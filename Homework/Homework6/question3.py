
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from data_preparation import X_train, X_val, y_train, y_val

# Define the number of trees to evaluate
n_estimators_values = [10, 50, 100, 150]

# Store model evaluation results
results = []

# Train and evaluate each random forest model
for n_estimators in n_estimators_values:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=1,
        n_jobs=-1
    )

    # Train the model
    model.fit(X_train, y_train)

    # Generate validation predictions
    y_pred = model.predict(X_val)

    # Calculate validation RMSE
    rmse = mean_squared_error(
        y_val,
        y_pred
    ) ** 0.5

    results.append({
        "n_estimators": n_estimators,
        "rmse": rmse,
        "rmse_3dp": round(rmse, 3)
    })

# Create a results DataFrame
results_df = pd.DataFrame(results)

# Select the best model using three decimal places
best_rmse = results_df["rmse_3dp"].min()

best_models = results_df[
    results_df["rmse_3dp"] == best_rmse
]

# Display results
print("\nQuestion 3 — Random Forest Hyperparameter Tuning")
print("-" * 55)

print("\nValidation RMSE comparison:")
print(results_df.to_string(index=False))

print("\nBest model(s):")
print(best_models.to_string(index=False))

# Save results for documentation
results_df.to_csv(
    "question3_results.csv",
    index=False
)

print("\nResults saved to question3_results.csv")
