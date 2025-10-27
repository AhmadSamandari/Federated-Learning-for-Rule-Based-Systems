import numpy as np
import re
import pandas as pd
import os
from sklearn.tree import _tree # Needed for parsing decision tree structure
from sklearn.tree import DecisionTreeClassifier
import sys
import pickle
from collections import defaultdict
import random # Added for relevance_weighted_sample

# --- Functions adapted from server_device_weighted.py (now consolidated) ---

# Function to parse the conditions from a string (adapted from server_device_weighted.py)
def parse_conditions(conditions_str):
    conditions = []
    # Use a more robust regex to split by ' and ' to ensure logical separation
    condition_parts = re.split(r'\s+and\s+', conditions_str.strip())

    for part in condition_parts:
        part = part.strip()
        if not part: # Skip empty parts
            continue
        
        # Regex to split on the first operator found (>=, <=, >, <, ==)
        # Ensure operators are correctly ordered for matching (e.g., >= before >)
        match = re.search(r'(>=|<=|>|<|==)', part)
        if match:
            feature = part[:match.start()].strip()
            operator = match.group(1)
            value = part[match.end():].strip()
            
            condition = {'feature': feature, 'operator': operator, 'value': value}
            conditions.append(condition)
    return conditions


# Function to load rules from file
def load_rules_from_file(file_path, min_coverage=3.0):
    rules = []
    if not os.path.exists(file_path):
        return rules

    with open(file_path, 'r') as file:
        current_rule_str = ""
        for line in file:
            line = line.strip()
            if line: # If line is not empty
                current_rule_str += line + " "
            if line.startswith('if') and 'relevance' in line: # End of a rule block heuristic
                # Now parse the rule string back into a dictionary
                # Example: "if N <= 5.0 and F0 > 10.0 class: 0 | coverage 80.0% | error 5.0% | relevance 76.0"
                match = re.search(r'if\s*(.*?)\s+class:\s*(\d+)\s*\|\s*coverage\s*(\d+\.?\d*)%\s*\|\s*error\s*(\d+\.?\d*)%\s*\|\s*relevance\s*(\d+\.?\d*)', current_rule_str, re.IGNORECASE)
                if match:
                    conditions_str = match.group(1).strip()
                    class_str = match.group(2)
                    coverage = float(match.group(3))
                    error_percentage = float(match.group(4))
                    relevance = float(match.group(5))

                    if coverage >= min_coverage:
                        conditions = parse_conditions(conditions_str)
                        class_prediction = int(class_str)
                        if conditions:
                            rule = {
                                'conditions': conditions,
                                'class': class_prediction,
                                'coverage': coverage,
                                'error_percentage': error_percentage,
                                'relevance': relevance
                            }
                            rules.append(rule)
                current_rule_str = "" # Reset for the next rule
            elif not line and current_rule_str: # If there's a blank line and we have a partial rule
                # This could happen if the file has inconsistent formatting, or a rule spans multiple lines
                # without an immediate 'relevance' at the end. For robustness, we'll try to parse
                # any accumulated string, but the above 'relevance' heuristic is stronger.
                # For now, let's assume one rule per 'if ... relevance' block.
                pass # The main logic is above.
    return rules


# Function to classify an instance using the given rules
def classify_instance(instance_dict, rules):
    for rule in rules:
        conditions = rule['conditions']
        all_conditions_satisfied = True # All conditions in a rule must be satisfied (AND logic)

        for condition in conditions:
            feature = condition['feature']
            operator = condition['operator']
            value_str = str(condition['value']).strip() # Ensure value is string for parsing

            if feature not in instance_dict:
                all_conditions_satisfied = False
                break # Rule cannot be applied if feature is missing

            instance_feature_value = instance_dict[feature]
            matched = False

            try:
                # Determine if categorical or numerical comparison
                # If instance value is string, or rule value is clearly non-numeric, treat as categorical
                if isinstance(instance_feature_value, str) or (isinstance(value_str, str) and not value_str.replace('.', '', 1).isdigit()):
                    matched = (str(instance_feature_value).strip() == value_str)
                else: # Assume numeric if not string
                    value_float = float(value_str)
                    if operator == '>':
                        matched = (instance_feature_value > value_float)
                    elif operator == '>=':
                        matched = (instance_feature_value >= value_float)
                    elif operator == '<':
                        matched = (instance_feature_value < value_float)
                    elif operator == '<=':
                        matched = (instance_feature_value <= value_float)
                    elif operator == '==':
                        # Using a tolerance for float comparisons
                        matched = np.isclose(instance_feature_value, value_float, atol=1e-6)
                    else: # Unrecognized operator
                        matched = False
            except ValueError:
                matched = False
            except TypeError:
                matched = False
            
            if not matched:
                all_conditions_satisfied = False
                break
            
        if all_conditions_satisfied:
            return rule['class']

    return None # No rule matched


# Function to predict the label for a point using the given rules (weighted vote)
def predict_label(instance_dict, merged_rules_list_of_lists, feature_names, target_label_names=None):
    
    if target_label_names:
        class_values = [int(label) for label in target_label_names]
    else:
        class_values = [0, 1]
        
    weighted_votes = {cls: 0.0 for cls in class_values}

    # Ensure instance_dict has numeric values for features that are expected to be numeric
    processed_instance_dict = {}
    for feature in feature_names:
        if feature in instance_dict:
            try:
                # Attempt to convert to float for features that should be numeric
                processed_instance_dict[feature] = float(instance_dict[feature])
            except (ValueError, TypeError):
                # Keep as string if it's not convertible (e.g., categorical)
                processed_instance_dict[feature] = instance_dict[feature]
        else:
            # Handle cases where a feature might be missing
            processed_instance_dict[feature] = None # Or raise an error, or use a default

    for rule_set in merged_rules_list_of_lists:
        for rule in rule_set:
            # Check if the current instance satisfies this specific rule's conditions
            # We classify using only this single rule to see if it 'fires' for the instance
            if classify_instance(processed_instance_dict, [rule]) == rule['class']:
                label = rule['class']
                weight = rule.get('relevance', 0.0) # Use relevance as the weight now

                if label in weighted_votes:
                    weighted_votes[label] += weight

    total_votes = sum(weighted_votes.values())
    if total_votes == 0:
        return None # No rules matched, or all matched rules had 0 relevance

    max_weight = -1
    majority_class = None
    
    # Sort keys to ensure deterministic tie-breaking if target_label_names provide an order
    sorted_class_values = sorted(weighted_votes.keys())

    for cls in sorted_class_values:
        if weighted_votes[cls] > max_weight:
            max_weight = weighted_votes[cls]
            majority_class = cls
            
    return majority_class


# Function to write merged rules to txt
def write_merged_rules_to_txt(file_path, merged_rules_list_of_lists):
    with open(file_path, 'w') as file:
        for rules_list in merged_rules_list_of_lists:
            for rule in rules_list:
                conditions_str = ' and '.join([f"{cond['feature']} {cond['operator']} {cond['value']}" for cond in rule['conditions']])
                file.write(f"if {conditions_str} class: {rule['class']} | coverage {rule['coverage']}% | error {rule['error_percentage']}% | relevance {rule['relevance']}\n\n")


# --- Original Utility for Decision Tree based rule extraction (adapted from your notebook) ---

def compute_error(rule_path_last_element, target_class_index, total_samples_all_other_classes):
    error_count = 0
    
    if not isinstance(rule_path_last_element, tuple) or len(rule_path_last_element) != 2:
        return 100.0 # Return maximum error if format is unexpected

    class_samples_array, _ = rule_path_last_element

    if class_samples_array.shape[0] == 1:
        class_counts = class_samples_array[0]
    else:
        class_counts = np.sum(class_samples_array, axis=0)

    if target_class_index < 0 or target_class_index >= len(class_counts):
        return 100.0 # Return maximum error if index is invalid

    for i, count in enumerate(class_counts):
        if i != target_class_index:
            error_count += count

    if total_samples_all_other_classes == 0:
        return 0.0 if error_count == 0 else 100.0
    
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
            p1, p2 = list(path), list(path)
            p1 += [f"{name} <= {np.round(threshold, 3)}"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"{name} > {np.round(threshold, 3)}"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # Sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    # Ensure target_label_names are integers for proper indexing if they represent labels
    unique_class_names = sorted(Y_target_series.unique())
    unique_class_names = [int(cn) for cn in unique_class_names] 
    
    num_classes = len(unique_class_names)
    total_samples_classes = [np.sum(Y_target_series == class_name) for class_name in unique_class_names]

    rules_list = []

    for path in paths:
        conditions_list_for_rule = []
        for p_cond in path[:-1]:
            conditions_list_for_rule.append(str(p_cond))

        # Apply group-specific conditions based on 'dataset_type' config
        dataset_type = config.get('dataset_type', 'HSP') # Default to HSP if not specified
        
        if dataset_type == 'HSP':
            if (group_id == 1):
                conditions_list_for_rule.append(f"age >= {config['age_group1_min']}")
                conditions_list_for_rule.append(f"age <= {config['age_group1_max']}")
            elif (group_id == 2):
                conditions_list_for_rule.append(f"age >= {config['age_group2_min']}")
                conditions_list_for_rule.append(f"age <= {config['age_group2_max']}")
        elif dataset_type == 'VP': # Using 'VP' for your new dataset type
            # Assuming 'm' is the grouping feature for VP
            # And min/max are defined in config similar to HSP
            if (group_id == 1):
                conditions_list_for_rule.append(f"m <= {config['m_group1_max']}")
            elif (group_id == 2):
                conditions_list_for_rule.append(f"m >= {config['m_group2_min']}")
        
        classes_counts_at_node = path[-1][0][0]
        l = np.argmax(classes_counts_at_node) # Predicted class index based on counts

        predicted_class_value = unique_class_names[l] # Get the actual class label (e.g., 0 or 1)
        samples_at_node = path[-1][1]
        
        total_predicted_class_samples_in_data = total_samples_classes[l]

        coverage = 0.0
        if total_predicted_class_samples_in_data > 0:
            coverage = np.round(100.0 * samples_at_node / total_predicted_class_samples_in_data, 2)
        
        # Calculate error percentage for this rule
        # The sum of all samples minus the samples of the *predicted* class within the entire dataset
        total_samples_all_other_classes = sum(total_samples_classes) - total_predicted_class_samples_in_data
        error_percentage = compute_error(path[-1], l, total_samples_all_other_classes)
        
        rule_relevance = coverage * (1 - error_percentage/100) # error_percentage is already a percent
        
        # Construct the rule dictionary
        rule_dict = {
            'conditions': parse_conditions(" and ".join(conditions_list_for_rule)), # Parse combined conditions
            'class': predicted_class_value,
            'coverage': coverage,
            'error_percentage': np.round(error_percentage, 2),
            'relevance': np.round(rule_relevance, 2)
        }
        rules_list.append(rule_dict)
        
    return rules_list

# Function to train a Decision Tree Classifier
def train_decision_tree_model(X_train_df, y_train_series):    
    # Using a fixed random_state for reproducibility as in your original script
    model = DecisionTreeClassifier(random_state=38).fit(X_train_df.values, y_train_series.values)
    return model

# Functions to get tree metrics and model size
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

# Function to extract boost centers (from your new code)
def extract_boost_centers(merged_rules, relevance_threshold=3.0):
    boost_centers = defaultdict(list)
    for rule_set in merged_rules:
        for rule in rule_set:
            # Check for 'relevance' key, or fallback to 'coverage' if relevance isn't present
            actual_relevance_metric = rule.get('relevance', rule.get('coverage', 0.0))
            if actual_relevance_metric >= relevance_threshold:
                for cond in rule['conditions']:
                    feat = cond['feature']
                    try:
                        val = float(cond['value'])
                        boost_centers[feat].append(val)
                    except ValueError: # Changed 'except' to specific ValueError
                        continue  # skip non-numeric values
    return boost_centers

# Function for relevance-weighted sampling (from your new code)
def relevance_weighted_sample(min_val, max_val, centers, p_bias=0.3, scale_ratio=0.05):
    if centers and random.random() < p_bias:
        center = random.choice(centers)
        std = scale_ratio * (max_val - min_val)
        return np.clip(np.random.normal(loc=center, scale=std), min_val, max_val)
    else:
        return random.uniform(min_val, max_val)

def add_laplace_noise(value, epsilon=0.5, sensitivity=1.0):
    """Add Laplace noise for DP."""
    scale = sensitivity / epsilon
    return value + np.random.laplace(0, scale)