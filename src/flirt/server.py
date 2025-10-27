import os
import shutil
import timeit
import random
from collections import deque

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from src.utils.rule_utils import (
    load_rules_from_file,
    predict_label,
    train_decision_tree_model,
    get_tree_metrics,
    get_model_size_bytes,
    write_merged_rules_to_txt,
    extract_boost_centers,
    relevance_weighted_sample,
)
from src.flirt.client import FlirtClient


class FlirtServer:
    """
    FlirtServer:
      - By default keeps rolling buffers for acc/f1/mcc (retain_last_n, default=10)
      - If you need arbitrary round-range averages (e.g., rounds 240-250),
        set config['record_full_history']=True or set retain_last_n >= desired window.
      - Tree size/depth metrics are optional: set config['compute_tree_metrics']=True to enable.
    """

    def __init__(self, config, global_test_data, feature_names, target_label_names):
        self.config = config
        self.global_test_data = global_test_data.copy()
        self.feature_names = feature_names
        self.target_label_names = target_label_names

        self.rules_output_folder = config.get('rules_output_folder', 'temp_client_rules')
        self.min_coverage_for_merge = config.get('min_coverage_for_merge', 3.0)
        self.synthetic_data_size = int(config.get('synthetic_data_size', 5000))
        self.server_val_split = float(config.get('server_val_split', 0.2))
        self.dataset_type = config.get('dataset_type', 'HSP')

        # behavior knobs
        self.retain_last_n = int(self.config.get('retain_last_n', 10))
        self.avg_last_n = int(self.config.get('avg_last_n', 10))
        self.avg_skip_zeros = bool(self.config.get('avg_skip_zeros', True))
        self.compute_tree_metrics = bool(self.config.get('compute_tree_metrics', False))
        self.record_full_history = bool(self.config.get('record_full_history', False))

        # seeds
        seed = int(self.config.get('random_state', 38))
        np.random.seed(seed)
        random.seed(seed)

        # Prepare test data and normalized labels
        self.X_global_test = self.global_test_data.iloc[:, :-1].reset_index(drop=True)
        self.y_global_test = self.global_test_data.iloc[:, -1].reset_index(drop=True)
        try:
            self.y_global_test = self.y_global_test.astype(float).astype(int)
        except Exception:
            pass

        # feature ranges for synthetic generation
        self.feature_min_values = self.X_global_test.min().to_dict()
        self.feature_max_values = self.X_global_test.max().to_dict()

        # histories: either rolling buffers (deques) or full lists
        if self.record_full_history:
            self.acc_history = []
            self.f1_history = []
            self.mcc_history = []
            if self.compute_tree_metrics:
                self.tree_sizes_history = []
                self.tree_depths_history = []
                self.model_sizes_history = []
        else:
            self.acc_history = deque(maxlen=self.retain_last_n)
            self.f1_history = deque(maxlen=self.retain_last_n)
            self.mcc_history = deque(maxlen=self.retain_last_n)
            if self.compute_tree_metrics:
                self.tree_sizes_history = deque(maxlen=self.retain_last_n)
                self.tree_depths_history = deque(maxlen=self.retain_last_n)
                self.model_sizes_history = deque(maxlen=self.retain_last_n)

        self.elapsed_time = 0.0

        # ensure rules folder exists (start clean)
        if os.path.exists(self.rules_output_folder):
            shutil.rmtree(self.rules_output_folder)
        os.makedirs(self.rules_output_folder, exist_ok=True)

        # ensure output folder exists
        out_dir = os.path.dirname(self.config.get('output_filename', 'results.npy'))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def _clear_rules_folder(self):
        if os.path.exists(self.rules_output_folder):
            for fn in os.listdir(self.rules_output_folder):
                fp = os.path.join(self.rules_output_folder, fn)
                try:
                    if os.path.isfile(fp):
                        os.remove(fp)
                except Exception:
                    pass

    def _generate_synthetic_data(self, boost_centers=None):
        boost_centers = boost_centers or {}
        p_bias = float(self.config.get('relevance_p_bias', 0.3))
        scale_ratio = float(self.config.get('relevance_scale_ratio', 0.05))

        rows = []
        for _ in range(self.synthetic_data_size):
            inst = {}
            for feat in self.feature_names:
                min_val = float(self.feature_min_values.get(feat, 0.0))
                max_val = float(self.feature_max_values.get(feat, 1.0))
                centers = boost_centers.get(feat, [])
                val = relevance_weighted_sample(min_val, max_val, centers, p_bias=p_bias, scale_ratio=scale_ratio)
                inst[feat] = val
            rows.append(inst)
        df = pd.DataFrame(rows, columns=self.feature_names)

        if self.dataset_type == 'VP':
            int_cols = self.config.get('vp_int_columns', [])
            for col in int_cols:
                if col in df.columns:
                    min_int = int(np.floor(self.feature_min_values.get(col, 0)))
                    max_int = int(np.ceil(self.feature_max_values.get(col, min_int + 1)))
                    if max_int <= min_int:
                        max_int = min_int + 1
                    df[col] = np.random.randint(min_int, max_int + 1, size=len(df))
        return df

    def _merge_and_label_synthetic_data(self, synthetic_data_df):
        if synthetic_data_df is None or synthetic_data_df.empty:
            return pd.DataFrame()

        if not os.path.exists(self.rules_output_folder) or not os.listdir(self.rules_output_folder):
            return pd.DataFrame()

        merged_rules = []
        for fn in os.listdir(self.rules_output_folder):
            if not fn.endswith('.txt'):
                continue
            fp = os.path.join(self.rules_output_folder, fn)
            rules = load_rules_from_file(fp, min_coverage=self.min_coverage_for_merge)
            if rules:
                merged_rules.append(rules)

        if not merged_rules:
            return pd.DataFrame()

        labels = []
        for _, row in synthetic_data_df.iterrows():
            inst = {f: row[f] for f in self.feature_names}
            pred = predict_label(inst, merged_rules, self.feature_names, target_label_names=self.target_label_names)
            labels.append(pred)

        labeled = synthetic_data_df.copy()
        labeled['labels'] = labels
        labeled.dropna(subset=['labels'], inplace=True)
        if labeled.empty:
            return pd.DataFrame()

        # coerce labels to ints when possible
        def to_int_safe(v):
            try:
                return int(float(v))
            except Exception:
                return v

        labeled['labels'] = labeled['labels'].apply(to_int_safe)
        try:
            labeled['labels'] = labeled['labels'].astype(int)
        except Exception:
            pass

        return labeled

    def _train_server_model(self, labeled_synthetic_data_df):
        X_train = labeled_synthetic_data_df.loc[:, self.feature_names]
        y_train = labeled_synthetic_data_df['labels']

        try:
            y_train = y_train.astype(int)
        except Exception:
            y_train = y_train.apply(lambda v: int(float(v)) if isinstance(v, (str, float)) else int(v))

        server_model = train_decision_tree_model(X_train, y_train)

        y_pred = server_model.predict(self.X_global_test.values)
        try:
            y_pred = np.asarray(y_pred).astype(int)
        except Exception:
            y_pred = [int(float(p)) if isinstance(p, (str, float)) else int(p) for p in y_pred]

        acc = float(accuracy_score(self.y_global_test, y_pred))
        f1 = float(f1_score(self.y_global_test, y_pred, average='weighted'))
        mcc = float(matthews_corrcoef(self.y_global_test, y_pred))

        # append metrics (either list or deque works with .append)
        self.acc_history.append(acc)
        self.f1_history.append(f1)
        self.mcc_history.append(mcc)

        if self.compute_tree_metrics:
            metrics = get_tree_metrics(server_model)
            model_size = get_model_size_bytes(server_model)
            self.tree_sizes_history.append(metrics.get('n_nodes', 0))
            self.tree_depths_history.append(metrics.get('max_depth', 0))
            self.model_sizes_history.append(model_size)

        return server_model

    def run_federated_training(self, train_client_batches_1, train_client_batches_2):
        total_g1 = len(train_client_batches_1)
        total_g2 = len(train_client_batches_2)
        total_available = total_g1 + total_g2

        print(f"Total global test samples: {len(self.X_global_test)}")
        print(f"Prepared {total_g1} batches for Group1 and {total_g2} for Group2 (total {total_available})")

        start_time = timeit.default_timer()

        num_rounds = int(self.config.get('num_rounds', 1))
        for r in range(1, num_rounds + 1):
            print(f"Round {r}/{num_rounds}")
            self._clear_rules_folder()

            requested = int(self.config.get('number_clients', total_available))
            requested = min(requested, total_available)

            num_g1 = requested // 2
            num_g2 = requested - num_g1
            num_g1 = min(num_g1, total_g1)
            num_g2 = min(num_g2, total_g2)
            if (num_g1 + num_g2) < requested:
                remaining = requested - (num_g1 + num_g2)
                if total_g1 - num_g1 >= remaining:
                    num_g1 += remaining
                elif total_g2 - num_g2 >= remaining:
                    num_g2 += remaining

            g1_indices = random.sample(list(range(total_g1)), k=num_g1) if num_g1 > 0 else []
            g2_indices = random.sample(list(range(total_g2)), k=num_g2) if num_g2 > 0 else []

            sampled_1 = [(i, train_client_batches_1[i]) for i in g1_indices]
            sampled_2 = [(i, train_client_batches_2[i]) for i in g2_indices]

            print(f"  sampled G1:{len(sampled_1)} G2:{len(sampled_2)}")

            # train clients and write rules
            for orig_idx, client_df in sampled_1:
                client = FlirtClient(client_id=f"G1_C{orig_idx+1}", config=self.config,
                                     feature_names=self.feature_names, target_label_names=self.target_label_names)
                client.train_and_extract_rules(client_df, group_id=1)

            for orig_idx, client_df in sampled_2:
                client = FlirtClient(client_id=f"G2_C{orig_idx+1}", config=self.config,
                                     feature_names=self.feature_names, target_label_names=self.target_label_names)
                client.train_and_extract_rules(client_df, group_id=2)

            # load rules for this round
            merged_rules = []
            if os.path.exists(self.rules_output_folder):
                for fn in os.listdir(self.rules_output_folder):
                    if not fn.endswith('.txt'):
                        continue
                    fp = os.path.join(self.rules_output_folder, fn)
                    rules = load_rules_from_file(fp, min_coverage=self.min_coverage_for_merge)
                    if rules:
                        merged_rules.append(rules)

            boost_centers = extract_boost_centers(merged_rules, relevance_threshold=self.config.get('relevance_threshold', 3.0))
            synthetic_df = self._generate_synthetic_data(boost_centers=boost_centers)
            labeled_df = self._merge_and_label_synthetic_data(synthetic_df)

            if labeled_df.empty:
                print("  No labeled synthetic data, appending zero placeholders and continuing.")
                # append zeros/placeholders to keep alignment
                self.acc_history.append(0.0)
                self.f1_history.append(0.0)
                self.mcc_history.append(0.0)
                if self.compute_tree_metrics:
                    self.tree_sizes_history.append(0)
                    self.tree_depths_history.append(0)
                    self.model_sizes_history.append(0)
                merged_rules_txt = os.path.join(self.rules_output_folder, f"merged_rules_round_{r}.txt")
                write_merged_rules_to_txt(merged_rules_txt, merged_rules)
                continue

            server_model = self._train_server_model(labeled_df)

            merged_rules_txt = os.path.join(self.rules_output_folder, f"merged_rules_round_{r}.txt")
            write_merged_rules_to_txt(merged_rules_txt, merged_rules)

        self.elapsed_time = timeit.default_timer() - start_time
        print(f"Federated training finished in {self.elapsed_time:.2f}s")

        # save minimal results (only histories kept)
        results = {
            'acc_history': list(self.acc_history) if not self.record_full_history else self.acc_history,
            'f1_history': list(self.f1_history) if not self.record_full_history else self.f1_history,
            'mcc_history': list(self.mcc_history) if not self.record_full_history else self.mcc_history,
            'elapsed_time': self.elapsed_time,
            'config': self.config
        }
        if self.compute_tree_metrics:
            results.update({
                'tree_sizes_history': list(self.tree_sizes_history) if not self.record_full_history else self.tree_sizes_history,
                'tree_depths_history': list(self.tree_depths_history) if not self.record_full_history else self.tree_depths_history,
                'model_sizes_history': list(self.model_sizes_history) if not self.record_full_history else self.model_sizes_history,
            })

        np.save(self.config.get('output_filename', 'results.npy'), results)
        print(f"Results saved to {self.config.get('output_filename', 'results.npy')}")

    # --- metrics helpers ---------------------------------------------------------

    def _avg_last(self, hist, n=None, skip_zeros=True):
        if n is None:
            n = self.avg_last_n
        vals = list(reversed(list(hist)))
        if skip_zeros:
            vals = [v for v in vals if v is not None and v != 0]
        if not vals:
            return 0.0
        use = vals[:max(1, min(n, len(vals)))]
        return float(sum(use)) / len(use)

    def print_final_metrics(self, last_n=None):
        """
        Print averages of Accuracy, F1, MCC over last_n rounds (default config avg_last_n).
        To get an arbitrary round range (e.g., 240-250) use get_avg_for_round_range(...)
        with record_full_history=True.
        """
        n = int(last_n) if last_n is not None else self.avg_last_n
        acc_avg = self._avg_last(self.acc_history, n, skip_zeros=self.avg_skip_zeros)
        f1_avg = self._avg_last(self.f1_history, n, skip_zeros=self.avg_skip_zeros)
        mcc_avg = self._avg_last(self.mcc_history, n, skip_zeros=self.avg_skip_zeros)

        print(f"Averages over last {n} rounds (skip_zeros={self.avg_skip_zeros}):")
        print(f"  Accuracy: {acc_avg:.6f}")
        print(f"  F1:       {f1_avg:.6f}")
        print(f"  MCC:      {mcc_avg:.6f}")

    def get_avg_for_round_range(self, from_round, to_round):
        """
        Return averages for specified inclusive round range [from_round, to_round].
        Rounds are 1-based. Requires record_full_history=True (full lists saved).
        """
        if not self.record_full_history:
            raise RuntimeError("get_avg_for_round_range requires config['record_full_history']=True "
                               "or increase retain_last_n to cover desired range.")
        if from_round < 1 or to_round < from_round:
            raise ValueError("Invalid from_round/to_round")
        # histories are lists where entry i corresponds to round i (1-based)
        # convert to 0-based indices
        start = from_round - 1
        end = to_round  # slice end is exclusive
        acc_slice = self.acc_history[start:end]
        f1_slice = self.f1_history[start:end]
        mcc_slice = self.mcc_history[start:end]

        def avg_safe(lst):
            if not lst:
                return 0.0
            vals = [v for v in lst if v is not None]
            return float(sum(vals)) / len(vals) if vals else 0.0

        return {
            'acc_avg': avg_safe(acc_slice),
            'f1_avg': avg_safe(f1_slice),
            'mcc_avg': avg_safe(mcc_slice),
            'n_rounds': len(acc_slice)
        }

    def save_results(self, output_filename=None):
        out = output_filename or self.config.get('output_filename', 'results.npy')
        # reuse save in run_federated_training; here save current kept histories
        results = {
            'acc_history': list(self.acc_history) if not self.record_full_history else self.acc_history,
            'f1_history': list(self.f1_history) if not self.record_full_history else self.f1_history,
            'mcc_history': list(self.mcc_history) if not self.record_full_history else self.mcc_history,
            'elapsed_time': self.elapsed_time,
            'config': self.config
        }
        if self.compute_tree_metrics:
            results.update({
                'tree_sizes_history': list(self.tree_sizes_history) if not self.record_full_history else self.tree_sizes_history,
                'tree_depths_history': list(self.tree_depths_history) if not self.record_full_history else self.tree_depths_history,
                'model_sizes_history': list(self.model_sizes_history) if not self.record_full_history else self.model_sizes_history,
            })
        try:
            np.save(out, results)
            print(f"Results saved to {out}")
        except Exception as e:
            print(f"Failed saving results to {out}: {e}")