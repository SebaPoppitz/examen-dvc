import os
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


# Define paths
DATA_PATH = "data/processed_data/"
MODEL_PATH = "models/"


def main():
    """Main function to execute the workflow."""
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("Performing grid search...")
    best_model, best_params = perform_grid_search(X_train, y_train, X_test, y_test)

    if best_model:
        print(f"Best model found: {best_model.__class__.__name__}")
        print(f"Best parameters: {best_params}")
        save_model(best_model)
    else:
        print("No suitable model was found.")


def load_data():
    """Loads the processed train and test datasets."""
    X_train = pd.read_csv(os.path.join(DATA_PATH, "X_train_normalized.csv"))
    X_test = pd.read_csv(os.path.join(DATA_PATH, "X_test_normalized.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv")).values.ravel()  # Convert to 1D array
    y_test = pd.read_csv(os.path.join(DATA_PATH, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test


def define_models():
    """Returns a dictionary of models and their respective hyperparameter grids."""
    return {
        "Ridge": {
            "model": Ridge(),
            "params": {"alpha": [0.1, 1, 10, 100]}
        },
        "Lasso": {
            "model": Lasso(),
            "params": {"alpha": [0.1, 1, 10, 100]}
        },
        "RandomForest": {
            "model": RandomForestRegressor(),
            "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
        },
        "SVR": {
            "model": SVR(),
            "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
        }
    }


def perform_grid_search(X_train, y_train, X_test, y_test):
    """
    Performs Grid Search on multiple regression models and selects the best one based on test Mean Squared Error (MSE).
    Returns the best model and its parameters.
    """
    models = define_models()
    best_model = None
    best_params = None
    best_score = float("inf")

    for model_name, config in models.items():
        print(f"Running Grid Search for {model_name}...")
        grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Evaluate the best model from Grid Search
        best_candidate = grid_search.best_estimator_
        y_pred = best_candidate.predict(X_test)
        test_score = mean_squared_error(y_test, y_pred)

        print(f"{model_name} best params: {grid_search.best_params_}, Test MSE: {test_score}")

        # Track the best performing model
        if test_score < best_score:
            best_score = test_score
            best_model = best_candidate
            best_params = grid_search.best_params_

    return best_model, best_params


def save_model(model):
    """Saves the best model as a pickle file."""
    os.makedirs(MODEL_PATH, exist_ok=True)  # Ensure model directory exists
    model_filename = os.path.join(MODEL_PATH, "best_params.pkl")

    with open(model_filename, "wb") as file:
        pickle.dump(model, file)

    print(f"Best model saved to {model_filename}")



if __name__ == "__main__":
    main()
