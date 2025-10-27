# src/fedprox/client.py

import tensorflow as tf
import numpy as np
from src.utils.model_utils import create_nn_model, compile_model_for_fedprox # We'll modify model_utils
from src.utils.data_utils import train_test_split_client, prepare_client_data


class FedProxClient:
    def __init__(self, client_id, input_dim, config):
        self.client_id = client_id
        self.input_dim = input_dim
        self.config = config
        # Create a new model for each client to ensure fresh weights based on global_weights
        self.model = create_nn_model(input_dim)
        # We need to compile with a specific loss function for the custom training loop
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def set_global_weights(self, global_weights):
        """Sets the client's model weights to the global weights."""
        self.model.set_weights(global_weights)

    def train_local_model(self, client_data, global_weights_for_prox):
        """Trains the client's model on local data using FedProx."""
        # Split client data into training and validation sets
        train_data, validation_data = train_test_split_client(
            client_data,
            test_size=self.config['client_val_split'],
            random_state=self.config['random_state']
        )
        X_train, y_train = prepare_client_data(train_data)
        X_val, y_val = prepare_client_data(validation_data)

        mu = self.config.get('mu', 0.0) # Get mu from config, default to 0 if not present
        num_epochs = self.config['local_epochs']
        batch_size = self.config['batch_size']

        train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

        # Use tf.data.Dataset for efficient batching
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)

        for epoch in range(num_epochs):
            train_loss_metric.reset_state()
            train_accuracy_metric.reset_state()

            for x_batch_train, y_batch_train in train_dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(x_batch_train, training=True)
                    loss = self.loss_object(y_batch_train, predictions)

                    # FedProx regularization term
                    if mu > 0 and global_weights_for_prox is not None:
                        prox_term = 0.0
                        for w, w_global in zip(self.model.trainable_variables, global_weights_for_prox):
                            # Ensure w_global is a tf.Variable or cast to tensor for subtraction
                            w_global_tensor = tf.constant(w_global, dtype=w.dtype)
                            prox_term += tf.reduce_sum(tf.square(w - w_global_tensor))
                        loss += (mu / 2) * prox_term

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                train_loss_metric(loss)
                train_accuracy_metric(y_batch_train, predictions)
        
        # Note: FedProx typically doesn't use a validation set during local training
        # for aggregation purposes, but rather for early stopping/monitoring.
        # We are only returning the trained weights here.
        return self.model.get_weights()