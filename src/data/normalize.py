import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define paths
DATA_PATH = "data/processed_data/"
X_TRAIN_FILE = os.path.join(DATA_PATH, "X_train.csv")
X_TEST_FILE = os.path.join(DATA_PATH, "X_test.csv")
X_TRAIN_NORM_FILE = os.path.join(DATA_PATH, "X_train_normalized.csv")
X_TEST_NORM_FILE = os.path.join(DATA_PATH, "X_test_normalized.csv")


def main():
    """Main function to execute the data normalization process."""
    print("Loading data...")
    X_train, X_test = load_data()

    print("Normalizing data...")
    X_train_normalized, X_test_normalized = normalize_data(X_train, X_test)

    print("Saving normalized data...")
    save_data(X_train_normalized, X_test_normalized)

    print("Normalization process completed successfully!")


def load_data():
    """Loads training and testing datasets."""
    X_train = pd.read_csv(X_TRAIN_FILE)
    X_test = pd.read_csv(X_TEST_FILE)
    return X_train, X_test

def normalize_data(X_train, X_test):
    """Normalizes feature sets using StandardScaler."""
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    # Convert back to DataFrames with original column names
    X_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns)
    X_test_normalized = pd.DataFrame(X_test_normalized, columns=X_test.columns)
    
    return X_train_normalized, X_test_normalized

def save_data(X_train_normalized, X_test_normalized):
    """Saves the normalized feature sets to CSV files."""
    os.makedirs(DATA_PATH, exist_ok=True)  # Ensure the directory exists

    X_train_normalized.to_csv(X_TRAIN_NORM_FILE, index=False)
    X_test_normalized.to_csv(X_TEST_NORM_FILE, index=False)

    print(f"Normalized data saved to {X_TRAIN_NORM_FILE} and {X_TEST_NORM_FILE}")


if __name__ == "__main__":
    main()
