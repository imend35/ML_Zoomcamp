
from sklearn.tree import DecisionTreeRegressor, export_text

from data_preparation import X_train, y_train, dv

# Train a decision tree with a maximum depth of 1
model = DecisionTreeRegressor(
    max_depth=1,
    random_state=1
)

model.fit(X_train, y_train)

# Retrieve the encoded feature names
feature_names = dv.get_feature_names_out()

# Identify the root split feature
root_feature_index = model.tree_.feature[0]
root_feature_name = feature_names[root_feature_index]

# Map one-hot encoded features to original columns
original_feature_name = root_feature_name.split("=")[0]

# Display the results
print("Question 1 — Decision Tree Regressor")
print("-" * 45)

print(f"Root split feature: {root_feature_name}")
print(f"Original feature: {original_feature_name}")

# Display the decision tree structure
print("\nDecision tree structure:")
print(
    export_text(
        model,
        feature_names=list(feature_names)
    )
)
