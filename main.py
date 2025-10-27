# main.py

import argparse
import yaml
import os
import tensorflow as tf

# Ensure paths are correct for imports
from src.utils.data_utils import load_and_split_data, split_data_into_batches
from src.fedavg.server import FedAvgServer
# from src.fedprox.server import FedProxServer # Uncomment when you add FedProx

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config.get('model_name', 'FedAvg')
    print(f"Running Federated Learning with {model_name} model...")

    # Set up random seed for reproducibility
    tf.random.set_seed(config.get('random_state', 42))
    np.random.seed(config.get('random_state', 42))
    random.seed(config.get('random_state', 42))


    # 1. Load and Split Data
    train_1, _, train_2, _, test_total = load_and_split_data(
        node1_path=config['dataset_node1_path'],
        node2_path=config['dataset_node2_path'],
        test_size=config['test_size'],
        random_state=config['random_state']
    )

    input_dim = train_1.shape[1] - 1

    # Split training data into client batches
    train_1_batches = split_data_into_batches(train_1, config['client_sample_size'])
    train_2_batches = split_data_into_batches(train_2, config['client_sample_size'])

    # 2. Initialize and Run Server
    if model_name == "FedAvg":
        server = FedAvgServer(input_dim=input_dim, global_test_data=test_total, config=config)
        server.run_federated_training(train_1_batches, train_2_batches)
    # elif model_name == "FedProx":
    #     server = FedProxServer(input_dim=input_dim, global_test_data=test_total, config=config)
    #     server.run_federated_training(train_1_batches, train_2_batches)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # 3. Report and Save Results
    server.print_final_metrics()
    model_summary = server.get_model_summary()
    print(f"Global Model Parameters: {model_summary['total_parameters']}")
    print(f"Global Model Memory (bytes): {model_summary['memory_bytes']}")

    output_dir = os.path.dirname(config['output_filename'])
    os.makedirs(output_dir, exist_ok=True)
    server.save_results(config['output_filename'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning experiments.")
    parser.add_argument('--config', type=str, default='configs/fedavg_config.yaml',
                        help='Path to the configuration YAML file.')
    args = parser.parse_args()
    main(args.config)