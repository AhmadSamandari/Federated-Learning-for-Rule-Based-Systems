import numpy as np
import random
import timeit
import tensorflow as tf # Import TensorFlow for the Input layer check
from src.utils.model_utils import create_nn_model, compile_model, evaluate_model, report_metric_summary
from src.utils.data_utils import prepare_client_data
from src.fedavg.client import FedAvgClient 

class FedAvgServer:
    def __init__(self, input_dim, global_test_data, config):
        self.input_dim = input_dim
        self.config = config
        # Pass layer_sizes from config to create_nn_model
        self.global_model = create_nn_model(self.input_dim, layer_sizes=config.get('nn_layer_sizes', [25, 60, 25]))
        self.global_model = compile_model(self.global_model, learning_rate=config['optimizer_learning_rate']) # Use optimizer_learning_rate
        self.global_weights = self.global_model.get_weights()

        # Handle the case where the global_test_data might have an 'Unnamed: 0' column
        if 'Unnamed: 0' in global_test_data.columns:
            global_test_data = global_test_data.drop(['Unnamed: 0'], axis=1)

        self.X_global_test, self.y_global_test = prepare_client_data(global_test_data)

        self.acc_history = []
        self.f1_history = []
        self.mcc_history = []
        self.elapsed_time = 0

    def aggregate_weights(self, client_weights_list):
        """Aggregates client weights using FedAvg."""
        aggregated_weights = []
        for i in range(len(self.global_weights)):
            layer_weights = [client_weights[i] for client_weights in client_weights_list]
            mean_weights = np.mean(layer_weights, axis=0)
            aggregated_weights.append(mean_weights)
        self.global_weights = aggregated_weights
        self.global_model.set_weights(self.global_weights)

    def run_federated_training(self, train_client_batches_1, train_client_batches_2):
        """Runs the federated learning process."""
        print(f"Total test samples: {len(self.X_global_test)}")
        # Note: Your notebook used a fixed sample size of 1500 for evaluation,
        # but your current server evaluates on the full test_total.
        # Keeping the current server's behavior as it's more standard.

        print(f"Number of total client batches: {len(train_client_batches_1) + len(train_client_batches_2)}")

        start_time = timeit.default_timer()

        for round_num in range(1, self.config['num_rounds'] + 1):
            print(f"Round {round_num}")

            # Sample clients for this round
            # Your notebook samples 12 from each, so 24 total.
            # config['number_clients'] should be set to 24 for VP dataset.
            num_clients_per_group = self.config['number_clients'] // 2
            sampled_clients_data_1 = random.sample(train_client_batches_1, num_clients_per_group)
            sampled_clients_data_2 = random.sample(train_client_batches_2, num_clients_per_group)
            sampled_clients_data = sampled_clients_data_1 + sampled_clients_data_2

            client_weights = []
            for i, client_data in enumerate(sampled_clients_data):
                client = FedAvgClient(client_id=f"client_{i}", input_dim=self.input_dim, config=self.config)
                # Set global weights before local training if it's not the first round
                # This logic is already correct
                if round_num > 1:
                    client.set_global_weights(self.global_weights)
                local_weights = client.train_local_model(client_data)
                client_weights.append(local_weights)

            self.aggregate_weights(client_weights)

            # Evaluate global model
            acc, f1, mcc = evaluate_model(self.global_model, self.X_global_test, self.y_global_test)
            self.acc_history.append(acc)
            self.f1_history.append(f1)
            self.mcc_history.append(mcc)
            print(f"Round {round_num} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}, MCC: {mcc:.4f}")


        self.elapsed_time = timeit.default_timer() - start_time

    def get_results(self):
        """Returns the training history and elapsed time."""
        return {
            'accuracy': self.acc_history,
            'f1_score': self.f1_history,
            'mcc': self.mcc_history,
            'elapsed_time': self.elapsed_time
        }

    def print_final_metrics(self):
        """Prints the final summary of metrics."""
        print(f"Total training time: {self.elapsed_time:.2f} seconds")
        report_metric_summary("Accuracy", self.acc_history)
        report_metric_summary("F1-score", self.f1_history)
        report_metric_summary("MCC", self.mcc_history)

    def save_results(self, output_filename):
        """Saves the results to a numpy file."""
        np.save(output_filename, [np.array(self.acc_history), np.array(self.f1_history), np.array(self.mcc_history)])
        print(f"Results saved to {output_filename}")

    def get_model_summary(self):
        """Returns the total parameters and memory required for the global model."""
        total_params = self.global_model.count_params()
        memory_bytes = total_params * 4  # Assuming float32
        return {"total_parameters": total_params, "memory_bytes": memory_bytes}