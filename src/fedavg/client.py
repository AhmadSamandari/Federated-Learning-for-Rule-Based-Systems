from src.utils.model_utils import create_nn_model, compile_model
from src.utils.data_utils import train_test_split_client, prepare_client_data
import numpy as np

class FedAvgClient:
    def __init__(self, client_id, input_dim, config):
        self.client_id = client_id
        self.input_dim = input_dim
        self.config = config
        # Pass layer_sizes from config to create_nn_model
        self.model = create_nn_model(self.input_dim, layer_sizes=config.get('nn_layer_sizes', [25, 60, 25]))
        self.model = compile_model(self.model, learning_rate=config['optimizer_learning_rate']) # Use optimizer_learning_rate

    def set_global_weights(self, global_weights):
        """Sets the client's model weights to the global weights."""
        self.model.set_weights(global_weights)

    def train_local_model(self, client_data):
        """Trains the client's model on local data."""
        # Handle 'Unnamed: 0' column if present in client_data
        if 'Unnamed: 0' in client_data.columns:
            client_data = client_data.drop(['Unnamed: 0'], axis=1)

        # Split client data into training and validation sets
        train_data, validation_data = train_test_split_client(
            client_data,
            test_size=self.config['client_val_split'],
            random_state=self.config['random_state']
        )
        X_train, y_train = prepare_client_data(train_data)
        X_val, y_val = prepare_client_data(validation_data)

        self.model.fit(
            X_train, y_train,
            epochs=self.config['local_epochs'],
            verbose=0,
            shuffle=False, # Shuffle handled by train_test_split_client if random_state is set
            validation_data=(X_val, y_val),
            batch_size=self.config['batch_size']
        )
        return self.model.get_weights()