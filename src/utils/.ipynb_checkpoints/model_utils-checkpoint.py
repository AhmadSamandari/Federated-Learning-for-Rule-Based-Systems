import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import numpy as np

def create_nn_model(input_dim, layer_sizes=[25, 60, 25]): # Added layer_sizes parameter
    """Creates a sequential neural network model with configurable layers."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),  # Explicit Input layer for clarity
    ] + [
        tf.keras.layers.Dense(size, activation='relu') for size in layer_sizes
    ] + [
        tf.keras.layers.Dense(1, activation='sigmoid') # Output layer
    ])
    return model

def compile_model(model, learning_rate=0.01):
    """Compiles the given Keras model."""
    # Note: Your notebook uses Adam, but your current model_utils uses SGD.
    # We should make this configurable via the config.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Changed to Adam
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['MeanSquaredError'])
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns accuracy, F1, and MCC scores."""
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)
    return acc, f1, mcc

def report_metric_summary(name, values):
    """Prints the average and standard deviation of a metric."""
    last10 = values[-10:] if len(values) >= 10 else values
    avg = np.mean(last10)
    std = np.std(last10)
    print(f"{name} (last 10 rounds): {avg:.4f} ± {std:.4f}")