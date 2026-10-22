
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

# Configuration
DATA_PATH = "car_fuel_efficiency_2026.csv"
TARGET = "fuel_efficiency_mpg"
RANDOM_STATE = 1

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Fill missing values with zeros
df = df.fillna(0)

# Split the dataset into training, validation, and test sets
df_full_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_STATE
)

df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=RANDOM_STATE
)

# Reset indices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Separate target variables
y_train = df_train[TARGET].values
y_val = df_val[TARGET].values
y_test = df_test[TARGET].values

# Remove target column from input features
X_train_df = df_train.drop(columns=[TARGET])
X_val_df = df_val.drop(columns=[TARGET])
X_test_df = df_test.drop(columns=[TARGET])

# Convert feature DataFrames to dictionaries
train_dicts = X_train_df.to_dict(orient="records")
val_dicts = X_val_df.to_dict(orient="records")
test_dicts = X_test_df.to_dict(orient="records")

# Fit the vectorizer on training data only
dv = DictVectorizer(sparse=True)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)

# Validate dataset preparation
assert len(df_train) == 6000
assert len(df_val) == 2000
assert len(df_test) == 2000

assert X_train.shape[0] == len(y_train)
assert X_val.shape[0] == len(y_val)
assert X_test.shape[0] == len(y_test)

assert TARGET not in dv.get_feature_names_out()

# Display dataset split information
print("Dataset preparation completed successfully.")

print("\nDataset splits:")
print(f"Training samples: {len(df_train)}")
print(f"Validation samples: {len(df_val)}")
print(f"Test samples: {len(df_test)}")

print("\nFeature matrices:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

print("\nTarget arrays:")
print(f"y_train: {y_train.shape}")
print(f"y_val: {y_val.shape}")
print(f"y_test: {y_test.shape}")

print("\nEncoded feature names:")
print(dv.get_feature_names_out().tolist())

print("\nMissing values after preparation:")
print(df.isnull().sum().sum())
