import numpy as np
import re

# Function to parse the conditions from a string
def parse_conditions(conditions_str):
    conditions = []
    condition_parts = conditions_str.split('and')

    for part in condition_parts:
        part = part.strip()
        feature, comparison = part.split(' ', 1)
        operator = comparison.split()[0]
        if len(comparison.split()) > 1:
            value = comparison.split()[1]
        else:
            value = None

        condition = {'feature': feature, 'operator': operator, 'value': value}
        conditions.append(condition)

    return conditions


def load_rules_from_file(file_path, min_coverage= 3.0):
    rules = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('if'):
                conditions_start_index = line.index('if') + len('if')
                class_start_index = line.index('class:') + len('class:')

                # Extract conditions and class information
                conditions_str = line[conditions_start_index:class_start_index].strip()
                conditions = parse_conditions(conditions_str)
                
                # Extract class information using regular expression
                class_match = re.search(r'class: (\d+)', line)
                class_str = class_match.group(1) if class_match else None


                # Extract coverage information using regular expression
                coverage_match = re.search(r'coverage (\d+\.\d+)', line)
                coverage = float(coverage_match.group(1)) if coverage_match else 0.0

                # Filter out rules with coverage lower than min_coverage
                if coverage >= min_coverage:
                    conditions = parse_conditions(conditions_str)
                    class_prediction = int(class_str) if class_str else None
                    rule = {'conditions': conditions, 'class': class_prediction, 'coverage': coverage}
                    rules.append(rule)
                    
    return rules



# Function to convert a NumPy array to a dictionary with named keys
def numpy_array_to_dict(instance, columns):
    return {column: value for column, value in zip(columns, instance)}


# Function to classify an instance using the given rules
def classify_instance(instance, rules, columns):
    # Convert the NumPy array to a dictionary with named keys
    instance_dict = {col: val for col, val in zip(columns, instance)}
    
    for rule in rules:
        conditions = rule['conditions']
        any_condition_satisfied = True  # Initialize to True for proper "OR" logic

        for condition in conditions:
            feature = condition['feature']
            operator = condition['operator']
            value = condition['value']

            if isinstance(instance_dict[feature], str):  # Check if the feature value is a string
                matched = (str(instance_dict[feature]) == value)
            else:
                if operator == '>':
                    matched = (instance_dict[feature] > float(value))
                elif operator == '>=':
                    matched = (instance_dict[feature] >= float(value))
                elif operator == '<':
                    matched = (instance_dict[feature] < float(value))
                elif operator == '<=':
                    matched = (instance_dict[feature] <= float(value))
                elif operator == '==':
                    matched = np.isclose(instance_dict[feature], float(value))

            any_condition_satisfied = any_condition_satisfied and matched
            
        if any_condition_satisfied:
            #print("Match")
            return rule['class']
        #else:
            #print("No match")

    return None


# Function to predict labels for an instance using the given rules
def predict_labels_for_instance(instance, merged_rules, columns):
    predictions = []
    for rule_set in merged_rules:
        class_prediction = classify_instance(instance, rule_set, columns)
        predictions.append(class_prediction)
    return predictions
    
# Function to predict the label for a point using the given rules
def predict_label(instance, merged_rules, columns):
    # Each rule set corresponds to a different client
    weighted_votes = {0: 0.0, 1: 0.0}

    for rule_set in merged_rules:
        for rule in rule_set:
            if classify_instance(instance, [rule],  columns) == rule['class']:
                label = rule['class']
                weight = rule.get('coverage', 0.0)  # use coverage as the weight
                if label in weighted_votes:
                    weighted_votes[label] += weight

    # Decide based on the total weighted vote
    if weighted_votes[0] > weighted_votes[1]:
        return 0
    elif weighted_votes[1] > weighted_votes[0]:
        return 1
    else:
        return None  # tie or no match

def write_merged_rules_to_txt(file_path, merged_rules):
    with open(file_path, 'w') as file:
        for rules_list in merged_rules:
            for rule in rules_list:
                conditions = rule['conditions']
                class_prediction = rule['class']
                coverage = rule.get('coverage', 0)  # Get coverage or default to 0

                # Convert conditions to a string representation
                conditions_str = ' and '.join([f"{cond['feature']} {cond['operator']} {cond['value']}" for cond in conditions])

                # Write the rule to the file with coverage and probability
                file.write(f"if {conditions_str} class: {class_prediction} | coverage {coverage}\n")
