
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from data_preparation import X_train, X_val, y_train, y_val

# Define hyperparameter values
max_depth_values = [10, 15, 20, 25]
n_estimators_values = [10, 50, 100, 150]

# Store individual model results
results = []

# Train and evaluate all hyperparameter combinations
for max_depth in max_depth_values:
    for n_estimators in n_estimators_values:

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=1,
            n_jobs=-1
        )

        # Train the random forest model
        model.fit(X_train, y_train)

        # Generate validation predictions
        y_pred = model.predict(X_val)

        # Calculate validation RMSE
        rmse = mean_squared_error(
            y_val,
            y_pred
        ) ** 0.5

        # Store the evaluation results
        results.append({
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "rmse": rmse
        })

        print(
            f"max_depth={max_depth}, "
            f"n_estimators={n_estimators}, "
            f"RMSE={rmse:.6f}"
        )

# Convert results into a DataFrame
results_df = pd.DataFrame(results)

# Calculate mean RMSE for each maximum depth
mean_rmse_df = (
    results_df
    .groupby("max_depth", as_index=False)["rmse"]
    .mean()
    .rename(columns={"rmse": "mean_rmse"})
)

# Identify the best maximum depth
best_row = mean_rmse_df.loc[
    mean_rmse_df["mean_rmse"].idxmin()
]

# Display the final comparison
print("\nQuestion 4 — Maximum Depth Optimization")
print("-" * 55)

print("\nMean validation RMSE by max_depth:")
print(mean_rmse_df.to_string(index=False))

print(
    f"\nBest max_depth: {int(best_row['max_depth'])}"
)
print(
    f"Best mean RMSE: {best_row['mean_rmse']:.6f}"
)

# Save detailed and aggregated results
results_df.to_csv(
    "question4_results.csv",
    index=False
)

mean_rmse_df.to_csv(
    "question4_mean_rmse.csv",
    index=False
)

print("\nResults saved successfully.")
