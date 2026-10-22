
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from data_preparation import X_train, y_train, dv

# Configure the random forest model
model = RandomForestRegressor(
    n_estimators=10,
    max_depth=20,
    random_state=1,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)

# Extract feature names and importance scores
feature_names = dv.get_feature_names_out()
importances = model.feature_importances_

# Create a feature importance DataFrame
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

# Sort features by importance
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)

# Display all feature importance scores
print("\nQuestion 5 — Feature Importance")
print("-" * 50)

print("\nAll feature importances:")
print(importance_df.to_string(index=False))

# Compare the four features from the assignment
candidate_features = [
    "vehicle_weight",
    "horsepower",
    "acceleration",
    "engine_displacement"
]

candidate_importances = importance_df[
    importance_df["feature"].isin(candidate_features)
].copy()

# Identify the most important candidate feature
best_feature = candidate_importances.iloc[0]

print("\nCandidate feature comparison:")
print(candidate_importances.to_string(index=False))

print("\nMost important feature:")
print(best_feature["feature"])

print(
    f"Importance score: {best_feature['importance']:.6f}"
)

# Save results
importance_df.to_csv(
    "question5_feature_importance.csv",
    index=False
)

print("\nResults saved successfully.")
