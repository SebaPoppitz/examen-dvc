import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
RAW_DATA_PATH = "data/raw_data/raw.csv"
PROCESSED_DATA_PATH = "data/processed_data/"


def main():
    """Main function to orchestrate data loading, processing, and saving."""
    print("Loading data...")
    df = load_data(RAW_DATA_PATH)

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Saving split data...")
    save_data(X_train, X_test, y_train, y_test, PROCESSED_DATA_PATH)

    print("Data splitting process completed successfully!")


def load_data(raw_data_path):
    """Loads the raw dataset from a CSV file."""
    df = pd.read_csv(raw_data_path)
    return df

def preprocess_data(df):
    """Drops unnecessary columns and splits features and target."""
    if 'date' in df.columns:
        df.drop(columns=['date'], inplace=True)  # Drop 'date' column if it exists
    
    X = df.drop(columns=['silica_concentrate'])
    y = df['silica_concentrate']
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_data(X_train, X_test, y_train, y_test, processed_data_path):
    """Saves the split data into separate CSV files."""
    os.makedirs(processed_data_path, exist_ok=True)  # Ensure the output directory exists

    X_train.to_csv(os.path.join(processed_data_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_data_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_data_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_data_path, "y_test.csv"), index=False)

    print(f"Data saved to {processed_data_path}")

if __name__ == "__main__":
    main()
