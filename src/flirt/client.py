# src/flirt/client.py

import os
import pandas as pd
from src.utils.rule_utils import train_decision_tree_model, get_rules_from_tree, write_merged_rules_to_txt

class FlirtClient:
    def __init__(self, client_id, config, feature_names, target_label_names):
        self.client_id = client_id
        self.config = config
        self.feature_names = feature_names
        self.target_label_names = target_label_names
        self.rules_folder = config['rules_output_folder']

    def train_and_extract_rules(self, client_data, group_id):
        X_client = client_data.iloc[:, :-1]
        y_client = client_data.iloc[:, -1]

        local_model = train_decision_tree_model(X_client, y_client)

        rules_list = get_rules_from_tree(
            local_model,
            self.feature_names,
            self.target_label_names,
            y_client,
            X_client,
            group_id,
            self.config # Pass the full config to rule extraction
        )

        os.makedirs(self.rules_folder, exist_ok=True)

        rules_filename = f"rules_client_{self.client_id}_group_{group_id}.txt"
        rules_file_path = os.path.join(self.rules_folder, rules_filename)

        # write_merged_rules_to_txt expects a list of lists, so we wrap rules_list
        write_merged_rules_to_txt(rules_file_path, [rules_list])

        return rules_file_path