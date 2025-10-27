import numpy as np
import random
import timeit
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import shutil
import yaml

from src.utils.rule_utils import (
    load_rules_from_file, predict_label,
    train_decision_tree_model, get_rules_from_tree,
    get_tree_metrics, get_model_size_bytes, write_merged_rules_to_txt
)
from src.utils.data_utils import prepare_client_data, split_data_into_batches, load_and_split_data
from src.flirt.client import FlirtClient

class FlirtServer:
    def __init__(self, config, global_test_data, feature_names, target_label_names):
        self.config = config
        self.global_test_data = global_test_data
        self.feature_names = feature_names
        self.target_label_names = target_label_names

        self.rules_output_folder = config['rules_output_folder']
        self.min_coverage_for_merge = config['min_coverage_for_merge']
        self.synthetic_data_size = config['synthetic_data_size']
        self.server_val_split = config['server_val_split']
        self.dataset_type = config.get('dataset_type', 'HSP') # Get dataset type from config

        self.acc_history = []
        self.f1_history = []
        self.mcc_history = []
        self.tree_sizes_history = []
        self.tree_depths_history = []
        self.tree_internal_nodes_history = []
        self.tree_leaf_nodes_history = []
        self.model_sizes_history = []
        self.elapsed_time = 0

        self.X_global_test = global_test_data.iloc[:, :-1]
        self.y_global_test = global_test_data.iloc[:, -1]

        # Calculate min/max for synthetic data generation from global test data
        # If your notebook used combined training data for min/max, consider passing that instead.
        self.feature_min_values = self.X_global_test.min().to_dict()
        self.feature_max_values = self.X_global_test.max().to_dict()

        # Ensure rules_output_folder exists and is clean at startup
        if os.path.exists(self.rules_output_folder):
            shutil.rmtree(self.rules_output_folder)
        os.makedirs(self.rules_output_folder, exist_ok=True)
        
        # Ensure results output folder exists
        os.makedirs(os.path.dirname(config['output_filename']), exist_ok=True)


    def _clear_rules_folder(self):
        # Clears all previously generated client rule files for a new round
        if os.path.exists(self.rules_output_folder):
            for filename in os.listdir(self.rules_output_folder):
                file_path = os.path.join(self.rules_output_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): # Should not happen, but for safety
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    def _generate_synthetic_data(self):
        points_data = []
        for j in range(self.synthetic_data_size):
            instance_values = {}
            for feature in self.feature_names:
                min_val = self.feature_min_values.get(feature, 0) # Default to 0 if not found
                max_val = self.feature_max_values.get(feature, 1) # Default to 1 if not found
                
                # Special handling for grouping features based on dataset type
                if self.dataset_type == 'HSP' and feature == 'age':
                    # Age distribution logic for HSP
                    if j % 2 == 0:
                        instance_values[feature] = random.randint(self.config['age_group1_min'], self.config['age_group1_max'])
                    else:
                        instance_values[feature] = random.randint(self.config['age_group2_min'], self.config['age_group2_max'])
                elif self.dataset_type == 'VP' and feature == 'm': # Grouping feature for VP dataset
                    if j % 2 == 0:
                        instance_values[feature] = random.uniform(min_val, self.config['m_group1_max'])
                    else:
                        instance_values[feature] = random.uniform(self.config['m_group2_min'], max_val)
                else:
                    # General random uniform for other features
                    instance_values[feature] = random.uniform(min_val, max_val)
            
            points_data.append(list(instance_values.values()))
        
        synthetic_data_df = pd.DataFrame(points_data, columns=self.feature_names)

        # Apply integer conversion for specific columns if dataset_type is VP, mirroring notebook logic
        if self.dataset_type == 'VP':
            int_columns = self.config.get('vp_int_columns', [])
            for col in int_columns:
                if col in synthetic_data_df.columns:
                    # Ensure the min/max for these columns are used for integer conversion
                    min_int = int(np.floor(self.feature_min_values.get(col, 0)))
                    max_int = int(np.ceil(self.feature_max_values.get(col, 1)))
                    synthetic_data_df[col] = np.random.randint(min_int, max_int + 1, len(synthetic_data_df))

        return synthetic_data_df

    def _merge_and_label_synthetic_data(self, synthetic_data_df):
        all_client_rule_sets = [] # This will be a list of lists of rules
        
        if not os.path.exists(self.rules_output_folder) or not os.listdir(self.rules_output_folder):
            print("Warning: No client rule files found to merge. Skipping synthetic data labeling.")
            return pd.DataFrame(columns=self.feature_names + ['labels']) # Return empty dataframe with columns

        for filename in os.listdir(self.rules_output_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.rules_output_folder, filename)
                rules = load_rules_from_file(file_path, self.min_coverage_for_merge)
                if rules:
                    all_client_rule_sets.append(rules)
            
        final_labels = []
        for index, instance_row in synthetic_data_df.iterrows():
            instance_dict = instance_row.to_dict() # Convert Series to dict
            label = predict_label(instance_dict, all_client_rule_sets, feature_names=self.feature_names, target_label_names=self.target_label_names)
            final_labels.append(label)
        
        labeled_data_df = synthetic_data_df.copy()
        labeled_data_df['labels'] = final_labels
        
        labeled_data_df.dropna(subset=['labels'], inplace=True)
        # Ensure labels are integers for classification models
        if not labeled_data_df.empty:
            labeled_data_df['labels'] = labeled_data_df['labels'].astype(int)

        return labeled_data_df

    def _train_server_model(self, labeled_synthetic_data_df):
        # Use a consistent random_state for reproducibility
        X_synthetic_train, y_synthetic_train = (
            labeled_synthetic_data_df.loc[:, self.feature_names],
            labeled_synthetic_data_df['labels']
        )
        
        server_model = train_decision_tree_model(X_synthetic_train, y_synthetic_train)
        return server_model, X_synthetic_train, y_synthetic_train

    def run_federated_training(self, train_client_batches_1, train_client_batches_2):
        print(f"Total global test samples: {len(self.X_global_test)}")
        num_total_client_batches = len(train_client_batches_1) + len(train_client_batches_2)
        print(f"Number of total client batches available: {num_total_client_batches}")

        start_time_total = timeit.default_timer()

        for round_num in range(1, self.config['num_rounds'] + 1):
            print(f"Round {round_num}")
            round_start_time = timeit.default_timer()

            self._clear_rules_folder() # Clear rules from previous round

            # Ensure we don't try to sample more clients than available batches
            num_clients_per_group = self.config['number_clients'] // 2
            
            # Handle cases where num_clients_per_group might be > number of available batches
            actual_sampled_clients_1 = min(num_clients_per_group, len(train_client_batches_1))
            actual_sampled_clients_2 = min(num_clients_per_group, len(train_client_batches_2))

            sampled_clients_data_1 = random.sample(train_client_batches_1, actual_sampled_clients_1)
            sampled_clients_data_2 = random.sample(train_client_batches_2, actual_sampled_clients_2)
            
            for i, client_data in enumerate(sampled_clients_data_1):
                client = FlirtClient(client_id=f"G1_C{i+1}", config=self.config,
                                        feature_names=self.feature_names, target_label_names=self.target_label_names)
                client.train_and_extract_rules(client_data, group_id=1)

            for i, client_data in enumerate(sampled_clients_data_2):
                client = FlirtClient(client_id=f"G2_C{i+1}", config=self.config,
                                        feature_names=self.feature_names, target_label_names=self.target_label_names)
                client.train_and_extract_rules(client_data, group_id=2)

            synthetic_data_df = self._generate_synthetic_data()
            labeled_synthetic_data_df = self._merge_and_label_synthetic_data(synthetic_data_df)
            
            if labeled_synthetic_data_df.empty:
                print(f"Warning: No labeled synthetic data generated in round {round_num}. Skipping server model training.")
                continue

            server_model, X_synthetic_train, y_synthetic_train = self._train_server_model(labeled_synthetic_data_df)

            # Store metrics for server model
            model_size = get_model_size_bytes(server_model)
            self.model_sizes_history.append(model_size)
            
            metrics = get_tree_metrics(server_model)
            self.tree_sizes_history.append(metrics["n_nodes"])
            self.tree_depths_history.append(metrics["max_depth"])
            self.tree_internal_nodes_history.append(metrics["n_internal_nodes"])
            self.tree_leaf_nodes_history.append(metrics["n_leaf_nodes"])

            # Evaluate on global test data
            y_pred = server_model.predict(self.X_global_test)
            acc = accuracy_score(self.y_global_test, y_pred)
            f1 = f1_score(self.y_global_test, y_pred, average="macro")
            mcc = matthews_corrcoef(self.y_global_test, y_pred)

            self.acc_history.append(acc)
            self.f1_history.append(f1)
            self.mcc_history.append(mcc)
            
            round_elapsed_time = timeit.default_timer() - round_start_time
            print(f"Round {round_num} completed in {round_elapsed_time:.2f} seconds.")
            print(f"Global Test Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, MCC: {mcc:.4f}")

        self.elapsed_time = timeit.default_timer() - start_time_total
        print(f"Federated training completed in {self.elapsed_time:.2f} seconds.")

        results = {
            'acc_history': self.acc_history,
            'f1_history': self.f1_history,
            'mcc_history': self.mcc_history,
            'tree_sizes_history': self.tree_sizes_history,
            'tree_depths_history': self.tree_depths_history,
            'tree_internal_nodes_history': self.tree_internal_nodes_history,
            'tree_leaf_nodes_history': self.tree_leaf_nodes_history,
            'model_sizes_history': self.model_sizes_history,
            'elapsed_time': self.elapsed_time,
            'config': self.config
        }
        np.save(self.config['output_filename'], results)
        print(f"Results saved to {self.config['output_filename']}")