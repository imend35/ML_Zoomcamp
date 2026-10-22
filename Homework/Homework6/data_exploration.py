
import pandas as pd

# Load the official 2026 fuel efficiency dataset
DATA_PATH = "car_fuel_efficiency_2026.csv"

df = pd.read_csv(DATA_PATH)

# Display dataset dimensions
print("Dataset shape:", df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Preview the first five records
print("\nFirst five rows:")
print(df.head())

# Inspect column types and non-null counts
print("\nDataset information:")
df.info()

# Identify missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display descriptive statistics
print("\nDescriptive statistics:")
print(df.describe(include="all"))

# Check the target variable
print("\nTarget variable:")
print(df["fuel_efficiency_mpg"].describe())
