import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(node1_path, node2_path, test_size=0.05, random_state=38):
    """
    Loads data from specified paths and splits it into training and testing sets
    for two nodes.

    Args:
        node1_path (str): Path to the CSV for node 1.
        node2_path (str): Path to the CSV for node 2.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generator.

    Returns:
        tuple: (train_1, test_1, train_2, test_2, test_total)
    """
    node1_df = pd.read_csv(node1_path)
    node2_df = pd.read_csv(node2_path)

    # Drop 'Unnamed: 0' if it exists in the raw loaded data
    if 'Unnamed: 0' in node1_df.columns:
        node1_df = node1_df.drop(['Unnamed: 0'], axis=1)
    if 'Unnamed: 0' in node2_df.columns:
        node2_df = node2_df.drop(['Unnamed: 0'], axis=1)

    train_1, test_1 = train_test_split(node1_df, test_size=test_size, shuffle=True, random_state=random_state)
    train_2, test_2 = train_test_split(node2_df, test_size=test_size, shuffle=True, random_state=random_state)

    test_total = pd.concat([test_1, test_2], axis=0)
    return train_1, test_1, train_2, test_2, test_total

def split_data_into_batches(data, batch_size):
    """Splits data into batches of specified size."""
    batches = [data.iloc[i:i + batch_size] for i in range(0, len(data), batch_size)]
    return batches

def train_test_split_client(data, test_size=0.1, shuffle=True, random_state=None):
    """Splits client data into training and validation sets."""
    if shuffle:
        data = data.sample(frac=1, random_state=random_state)
    split_index = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_index]
    validation_data = data.iloc[split_index:]
    return train_data, validation_data

def prepare_client_data(client_df):
    """Extracts features and labels from a client DataFrame."""
    # This function is called after data splitting/batching,
    # the 'Unnamed: 0' column should ideally be removed earlier
    # or handle it here if it somehow persists.
    if 'Unnamed: 0' in client_df.columns:
        client_df = client_df.drop(['Unnamed: 0'], axis=1)

    X = client_df.iloc[:, :-1].values.astype('float32')
    y = client_df.iloc[:, -1].values.astype('float32')
    return X, y