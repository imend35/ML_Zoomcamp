
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from data_preparation import X_train, X_val, y_train, y_val

# Configure the random forest regressor
model = RandomForestRegressor(
    n_estimators=10,
    random_state=1,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)

# Generate predictions on the validation set
y_pred = model.predict(X_val)

# Calculate validation RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

# Display the evaluation results
print("\nQuestion 2 — Random Forest Regressor")
print("-" * 45)
print(f"Number of trees: {model.n_estimators}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Validation RMSE: {rmse:.6f}")
print(f"Validation RMSE (4 decimals): {rmse:.4f}")
