import os
import json
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Define paths
MODEL_PATH = "../../models/"
DATA_PATH = "../../data/processed_data/"
PREDICTIONS_PATH = "../../data/"
METRICS_PATH = "../../metrics/"

PREDICTIONS_FILENAME = os.path.join(PREDICTIONS_PATH, "predictions.csv")
SCORES_FILENAME = os.path.join(METRICS_PATH, "scores.json")



def main():
    """Main function to execute the evaluation pipeline."""
    print("Loading test data...")
    X_test, y_test = load_data()

    print("Loading trained model...")
    model = load_trained_model()

    print("Generating predictions...")
    predictions = generate_predictions(model, X_test)

    print("Calculating model performance metrics...")
    metrics = evaluate_model(y_test, predictions)

    print("Evaluation completed!")
    print(json.dumps(metrics, indent=4))



def load_data():
    """Loads the processed test dataset."""
    X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test_normalized.csv"))
    y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv")).values.ravel()  # Convert to 1D array
    return X_test, y_test


def load_trained_model():
    """Loads the trained model from a pickle file."""
    trained_model_path = os.path.join(MODEL_PATH, "gbr_model.pkl")
    if not os.path.exists(trained_model_path):
        raise FileNotFoundError(f"Trained model file not found at {trained_model_path}")

    with open(trained_model_path, "rb") as file:
        model = pickle.load(file)

    print(f"Trained model loaded: {model.__class__.__name__}")
    return model


def generate_predictions(model, X_test):
    """Generates predictions using the trained model."""
    print("Generating predictions...")
    predictions = model.predict(X_test)
    
    # Convert to DataFrame and save
    predictions_df = pd.DataFrame(predictions, columns=["predicted_silica_concentrate"])
    os.makedirs(PREDICTIONS_PATH, exist_ok=True)  # Ensure directory exists
    predictions_df.to_csv(PREDICTIONS_FILENAME, index=False)

    print(f"Predictions saved to {PREDICTIONS_FILENAME}")
    return predictions


def evaluate_model(y_test, predictions):
    """Calculates model performance metrics and saves them."""
    print("Evaluating model performance...")
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Save results as JSON
    metrics = {"Mean Squared Error": mse, "R2 Score": r2}
    os.makedirs(METRICS_PATH, exist_ok=True)  # Ensure directory exists
    with open(SCORES_FILENAME, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Performance metrics saved to {SCORES_FILENAME}")
    return metrics



if __name__ == "__main__":
    main()
