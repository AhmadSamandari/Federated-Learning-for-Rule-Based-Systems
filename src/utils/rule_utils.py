import os
import re
import sys
import pickle
import random
import numpy as np
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict

# --- Functions adapted from server_device_weighted.py (now consolidated) ---

def parse_conditions(conditions_str):
    conditions = []
    if not conditions_str:
        return conditions
    condition_parts = re.split(r'\s+and\s+', conditions_str.strip())
    for part in condition_parts:
        part = part.strip()
        if not part:
            continue
        # split on first operator
        m = re.search(r'(>=|<=|==|>|<)', part)
        if not m:
            continue
        op = m.group(1)
        left = part[:m.start(1)].strip()
        right = part[m.end(1):].strip()
        # try to convert right to float/int
        try:
            if '.' in right:
                right_val = float(right)
            else:
                right_val = int(right)
        except Exception:
            right_val = right
        conditions.append((left, op, right_val))
    return conditions

def load_rules_from_file(file_path, min_coverage=3.0):
    rules = []
    if not os.path.exists(file_path):
        return rules
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            # basic parsing: assume "rule\tcoverage\tclass"
            parts = line.split('\t')
            try:
                rule_text = parts[0]
                coverage = float(parts[1]) if len(parts) > 1 else 0.0
                cls = parts[2] if len(parts) > 2 else None
            except Exception:
                continue
            if coverage < min_coverage:
                continue
            rules.append({'rule': rule_text, 'coverage': coverage, 'class': cls})
    return rules

def classify_instance(instance_dict, rules):
    for rule in rules:
        conds = parse_conditions(rule['rule'])
        match = True
        for (feat, op, val) in conds:
            inst_val = instance_dict.get(feat, None)
            if inst_val is None:
                match = False
                break
            try:
                if op == '>':
                    if not (inst_val > val):
                        match = False; break
                elif op == '<':
                    if not (inst_val < val):
                        match = False; break
                elif op == '>=':
                    if not (inst_val >= val):
                        match = False; break
                elif op == '<=':
                    if not (inst_val <= val):
                        match = False; break
                elif op == '==':
                    if not (inst_val == val):
                        match = False; break
                else:
                    match = False; break
            except Exception:
                match = False
                break
        if match:
            return rule.get('class')
    return None

def predict_label(instance_dict, merged_rules_list_of_lists, feature_names, target_label_names=None):
    if target_label_names:
        class_values = list(target_label_names)
    else:
        # infer class values from merged rules
        class_values = set()
        for rule_set in merged_rules_list_of_lists:
            for rule in rule_set:
                if rule.get('class') is not None:
                    class_values.add(rule['class'])
        class_values = sorted(list(class_values))

    weighted_votes = {cls: 0.0 for cls in class_values}
    processed_instance_dict = {}
    for feature in feature_names:
        processed_instance_dict[feature] = instance_dict.get(feature, instance_dict.get(str(feature), None))

    for rule_set in merged_rules_list_of_lists:
        for rule in rule_set:
            cls = rule.get('class')
            if cls is None:
                continue
            cov = float(rule.get('coverage', 1.0))
            # simple vote: add coverage as weight
            # If rule matches instance, add weight
            conds = parse_conditions(rule['rule'])
            match = True
            for (feat, op, val) in conds:
                inst_val = processed_instance_dict.get(feat)
                if inst_val is None:
                    match = False; break
                try:
                    if op == '>':
                        if not (inst_val > val): match = False; break
                    elif op == '<':
                        if not (inst_val < val): match = False; break
                    elif op == '>=':
                        if not (inst_val >= val): match = False; break
                    elif op == '<=':
                        if not (inst_val <= val): match = False; break
                    elif op == '==':
                        if not (inst_val == val): match = False; break
                except Exception:
                    match = False; break
            if match:
                weighted_votes[cls] = weighted_votes.get(cls, 0.0) + cov

    total_votes = sum(weighted_votes.values())
    if total_votes == 0:
        return None

    # deterministic tie-break: sorted classes
    sorted_class_values = sorted(weighted_votes.keys())
    majority_class = max(sorted_class_values, key=lambda c: (weighted_votes[c], -sorted_class_values.index(c)))

    # Normalize return type: try to return an int if possible, otherwise original value
    if majority_class is None:
        return None
    try:
        return int(float(majority_class))
    except Exception:
        return majority_class

def write_merged_rules_to_txt(file_path, merged_rules_list_of_lists):
    with open(file_path, 'w') as file:
        for rule_set in merged_rules_list_of_lists:
            for r in rule_set:
                line = f"{r.get('rule','')}\t{r.get('coverage',0.0)}\t{r.get('class', '')}\n"
                file.write(line)

def compute_error(rule_path_last_element, target_class_index, total_samples_all_other_classes):
    error_count = 0
    if not isinstance(rule_path_last_element, tuple) or len(rule_path_last_element) != 2:
        return 100.0
    class_samples_array, _ = rule_path_last_element
    if class_samples_array.shape[0] == 1:
        return 0.0
    else:
        class_counts = class_samples_array
    if target_class_index < 0 or target_class_index >= len(class_counts):
        return 100.0
    for i, count in enumerate(class_counts):
        if i == target_class_index:
            continue
        error_count += count
    if total_samples_all_other_classes == 0:
        return 100.0
    error_percentage = (error_count / total_samples_all_other_classes) * 100
    return error_percentage

def get_rules_from_tree(tree_model, feature_names, target_label_names, Y_target_series, X_train_df, group_id, config):
    tree_ = tree_model.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature]

    paths = []
    path = []

    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # left
            left_path = path.copy()
            left_path.append((name, '<=', threshold))
            recurse(tree_.children_left[node], left_path, paths)
            # right
            right_path = path.copy()
            right_path.append((name, '>', threshold))
            recurse(tree_.children_right[node], right_path, paths)
        else:
            # leaf
            value = tree_.value[node][0]
            samples = int(tree_.n_node_samples[node]) if hasattr(tree_, 'n_node_samples') else int(sum(value))
            # determine majority class in leaf
            class_idx = int(np.argmax(value))
            class_name = sorted(Y_target_series.unique())[class_idx]
            rules_repr = " and ".join([f"{f} {op} {val}" for (f, op, val) in path])
            paths.append((path, (value, samples, class_name)))

    recurse(0, path, paths)

    # Sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    unique_class_names = sorted(Y_target_series.unique())
    try:
        unique_class_names = [int(cn) for cn in unique_class_names]
    except Exception:
        pass

    rules_list = []
    for path in paths:
        path_conditions = path[0]
        last_info = path[1]
        value_arr, samples, class_name = last_info
        rule_str = " and ".join([f"{f} {op} {val}" for (f, op, val) in path_conditions])
        coverage = float(samples)
        rules_list.append({'rule': rule_str, 'coverage': coverage, 'class': class_name})
    return rules_list

def train_decision_tree_model(X_train_df, y_train_series):
    model = DecisionTreeClassifier(random_state=38)
    model.fit(X_train_df.values, y_train_series.values)
    return model

def get_tree_metrics(tree):
    n_nodes = tree.tree_.node_count
    n_leaf_nodes = sum(tree.tree_.children_left == -1)
    n_internal_nodes = n_nodes - n_leaf_nodes
    max_depth = tree.tree_.max_depth
    return {
        "n_nodes": n_nodes,
        "n_leaf_nodes": n_leaf_nodes,
        "n_internal_nodes": n_internal_nodes,
        "max_depth": max_depth,
    }

def get_model_size_bytes(model):
    return sys.getsizeof(pickle.dumps(model))

def extract_boost_centers(merged_rules, relevance_threshold=3.0):
    boost_centers = defaultdict(list)
    for rule_set in merged_rules:
        for rule in rule_set:
            cov = float(rule.get('coverage', 0.0))
            if cov < relevance_threshold:
                continue
            conds = parse_conditions(rule.get('rule', ''))
            for (feat, op, val) in conds:
                if isinstance(val, (int, float)):
                    boost_centers[feat].append(float(val))
    # reduce to unique centers
    for k in list(boost_centers.keys()):
        vals = sorted(set(boost_centers[k]))
        boost_centers[k] = vals
    return boost_centers

def relevance_weighted_sample(min_val, max_val, centers, p_bias=0.3, scale_ratio=0.05):
    if centers and random.random() < p_bias:
        # pick a center then sample near it
        center = random.choice(centers)
        scale = (max_val - min_val) * scale_ratio
        return float(np.clip(np.random.normal(loc=center, scale=scale), min_val, max_val))
    else:
        return float(np.random.uniform(min_val, max_val))

