import pandas as pd
from sklearn.preprocessing import StandardScaler


X_train_path = "../../data/processed_data/X_train.csv"
X_test_path = "../../data/processed_data/X_test.csv"

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)


# Normalize the feature sets
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Convert normalized arrays back to DataFrames
X_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns)
X_test_normalized = pd.DataFrame(X_test_normalized, columns=X_test.columns)

# Save the normalized feature sets
X_train_normalized.to_csv("../../data/processed_data/X_train_normalized.csv", index=False)
X_test_normalized.to_csv("../../data/processed_data/X_test_normalized.csv", index=False)
