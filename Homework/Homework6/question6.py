
import numpy as np
import pandas as pd
import xgboost as xgb

from data_preparation import X_train, X_val, y_train, y_val

# Create XGBoost data matrices
dtrain = xgb.DMatrix(
    X_train,
    label=y_train
)

dval = xgb.DMatrix(
    X_val,
    label=y_val
)

# Define evaluation datasets
watchlist = [
    (dtrain, "train"),
    (dval, "val")
]

# Define common model parameters
base_params = {
    "max_depth": 6,
    "min_child_weight": 1,
    "objective": "reg:squarederror",
    "nthread": 8,
    "seed": 1,
    "verbosity": 1
}

# Define learning rates to compare
eta_values = [0.3, 0.1]

# Store evaluation results
results = []

for eta in eta_values:

    # Configure the learning rate
    params = base_params.copy()
    params["eta"] = eta

    # Store evaluation history
    evals_result = {}

    # Train the XGBoost model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=watchlist,
        evals_result=evals_result,
        verbose_eval=False
    )

    # Generate validation predictions
    y_pred = model.predict(dval)

    # Calculate validation RMSE
    rmse = np.sqrt(
        np.mean((y_val - y_pred) ** 2)
    )

    # Record final training and validation metrics
    results.append({
        "eta": eta,
        "train_rmse": evals_result["train"]["rmse"][-1],
        "validation_rmse": rmse,
        "logged_validation_rmse": evals_result["val"]["rmse"][-1]
    })

    print(f"\nXGBoost model with eta={eta}")
    print(f"Validation RMSE: {rmse:.6f}")

# Compare both learning rates
results_df = pd.DataFrame(results)

best_row = results_df.loc[
    results_df["validation_rmse"].idxmin()
]

print("\nQuestion 6 — XGBoost Learning Rate Comparison")
print("-" * 55)

print("\nModel comparison:")
print(results_df.to_string(index=False))

print(f"\nBest eta: {best_row['eta']}")
print(
    f"Best validation RMSE: "
    f"{best_row['validation_rmse']:.6f}"
)

# Save evaluation results
results_df.to_csv(
    "question6_results.csv",
    index=False
)

print("\nResults saved to question6_results.csv")
