import os
import pandas as pd
from sklearn.model_selection import train_test_split

raw_data_relative_path = "../../data/raw_data/raw.csv"
split_data_relative_path = "../../data/processed_data/"


df = pd.read_csv(raw_data_relative_path)

# Drop date col
df.drop(columns = ['date'], inplace=True)

#Split the data into feats and target
X = df.drop(columns = ['silica_concentrate'])
y = df['silica_concentrate']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Save the split data to CSV files
X_train.to_csv(os.path.join(split_data_relative_path, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(split_data_relative_path, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(split_data_relative_path, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(split_data_relative_path, "y_test.csv"), index=False)