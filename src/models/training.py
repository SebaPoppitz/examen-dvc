import os
import pickle
import pandas as pd

# Define model path
MODEL_PATH = "models/"
DATA_PATH = "data/processed_data/"
TRAINED_MODEL_FILENAME = os.path.join(MODEL_PATH, "gbr_model.pkl")

def main():
    """Main function to execute the training pipeline."""
    print("Loading training data...")
    X_train, y_train = load_data()

    print("Loading best model...")
    model = load_best_model()

    print("Training the model...")
    trained_model = train_model(model, X_train, y_train)

    print("Saving the trained model...")
    save_trained_model(trained_model)

def load_data():
    """Loads the processed train dataset."""
    X_train = pd.read_csv(os.path.join(DATA_PATH, "X_train_normalized.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv")).values.ravel()  # Convert to 1D array
    return X_train, y_train

def load_best_model():
    """Loads the best model from a pickle file."""
    best_model_path = os.path.join(MODEL_PATH, "best_params.pkl")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model file not found at {best_model_path}")

    with open(best_model_path, "rb") as file:
        model = pickle.load(file)
    
    print(f"Best model loaded: {model.__class__.__name__}")
    return model

def train_model(model, X_train, y_train):
    """Trains the model using the provided training data."""
    print("Training model...")
    model.fit(X_train, y_train)
    return model

def save_trained_model(model):
    """Saves the trained model as a pickle file."""
    os.makedirs(MODEL_PATH, exist_ok=True)  # Ensure the model directory exists
    with open(TRAINED_MODEL_FILENAME, "wb") as file:
        pickle.dump(model, file)

    print(f"Trained model saved to {TRAINED_MODEL_FILENAME}")


if __name__ == "__main__":
    main()
