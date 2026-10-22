## Question 1 — Decision Tree Regressor

**Answer: `model_year`**

A DecisionTreeRegressor was trained with:

- `max_depth=1`
- `random_state=1`

The root split was based on `model_year` with a threshold of 1997.5.

Predicted values:

- Model year <= 1997.5: 28.61 MPG
- Model year > 1997.5: 31.17 MPG

The result confirms that `model_year` is the first splitting feature of the trained decision tree.

## Question 2 — Random Forest Regressor

**Answer: `1.837`**

### Objective

Train a Random Forest Regressor to predict vehicle fuel efficiency (`fuel_efficiency_mpg`) and evaluate its performance using Root Mean Squared Error (RMSE) on the validation dataset.

### Model Configuration

| Parameter          | Value                 |
| ------------------ | --------------------- |
| Algorithm          | RandomForestRegressor |
| n_estimators       | 10                    |
| random_state       | 1                     |
| n_jobs             | -1                    |
| Training samples   | 6,000                 |
| Validation samples | 2,000                 |
| Evaluation metric  | RMSE                  |

### Evaluation Results

The same dataset and model configuration were evaluated in two Python environments.

| scikit-learn Version | Validation RMSE |
| -------------------- | --------------: |
| 1.9.0                |        1.833220 |
| **1.6.1**            |    **1.837054** |

### Reproducibility Investigation

The initial experiment using scikit-learn 1.9.0 produced a validation RMSE of 1.833220, which did not exactly match the provided answer options.

To investigate the discrepancy:

1. The local dataset was compared with the official 2026 dataset.
2. Both datasets were confirmed to contain identical values.
3. The preprocessing steps and Random Forest parameters were checked against the official assignment.
4. An isolated Python environment was created with scikit-learn 1.6.1.
5. The model was retrained using the same dataset and configuration.

The second experiment produced a validation RMSE of **1.837054**, matching the provided answer **1.837** when rounded to three decimal places.

The results demonstrate that dependency versions can affect numerical reproducibility. Because other dependencies also differed between environments, the observed difference cannot yet be attributed exclusively to scikit-learn.

### Conclusion

The verified multiple-choice answer is **`1.837`**, supported by the validation RMSE of **1.837054** obtained in the scikit-learn 1.6.1 environment.

## Question 3 — Random Forest Hyperparameter Tuning

**Answer: `n_estimators = 150`**

Four random forest models were evaluated using
different numbers of trees.

| n_estimators | Validation RMSE |
| -----------: | --------------: |
|           10 |        1.833220 |
|           50 |        1.773988 |
|          100 |        1.769786 |
|          150 |        1.769198 |

The random forest with 150 trees achieved the
lowest validation RMSE of 1.769 (rounded to
three decimal places).

Increasing the number of trees improved
validation performance, although the gains
became smaller beyond 50 trees.

## Question 4 — Random Forest Maximum Depth Optimization

**Answer: `max_depth = 10`**

### Objective

Identify the optimal maximum tree depth for a Random Forest Regressor by comparing average validation RMSE across multiple hyperparameter combinations.

### Hyperparameter Configuration

The following values were evaluated:

- `max_depth`: 10, 15, 20, 25
- `n_estimators`: 10, 50, 100, 150
- `random_state`: 1
- `n_jobs`: -1

A total of **16 Random Forest models** were trained and evaluated.

### Evaluation Results

For each maximum depth, the validation RMSE was calculated for four different numbers of trees. The mean RMSE was then computed for each depth.

| max_depth | Mean Validation RMSE |
| --------- | -------------------: |
| **10**    |         **1.754392** |
| 15        |             1.784224 |
| 20        |             1.786771 |
| 25        |             1.786305 |

### Key Findings

- A maximum depth of 10 achieved the lowest average validation RMSE.
- Increasing the maximum depth beyond 10 did not improve average validation performance.
- Deeper trees may capture more complex patterns but can also increase the risk of overfitting.
- The results demonstrate the importance of hyperparameter tuning when optimizing ensemble models.

### Conclusion

The best maximum tree depth was **`max_depth = 10`**, achieving a mean validation RMSE of **1.754392 MPG** across the four evaluated forest sizes.

This configuration provided the best average validation performance among the tested depth values.

## Question 5 — Random Forest Feature Importance

**Answer: `vehicle_weight`**

### Objective

Identify the most important feature among the four candidate variables using a trained Random Forest Regressor.

### Model Configuration

| Parameter    | Value                 |
| ------------ | --------------------- |
| Algorithm    | RandomForestRegressor |
| n_estimators | 10                    |
| max_depth    | 20                    |
| random_state | 1                     |
| n_jobs       | -1                    |

### Feature Importance Results

The model's `feature_importances_` attribute was used to calculate feature importance scores.

| Candidate Feature   | Importance Score |
| ------------------- | ---------------: |
| **vehicle_weight**  |     **0.203313** |
| horsepower          |         0.078271 |
| engine_displacement |         0.061281 |
| acceleration        |         0.054998 |

### Key Findings

- `vehicle_weight` achieved the highest importance score among the four candidate features.
- `horsepower` ranked second, followed by `engine_displacement` and `acceleration`.
- Across all 16 encoded features, `model_year` had the highest overall importance score (0.296107).
- Feature importance scores describe the relative contribution of features to impurity reduction within the trained forest; they do not establish causation.

### Conclusion

The correct answer is **`vehicle_weight`**, with a feature importance score of **0.203313**.

The complete feature importance results are available in `question5_feature_importance.csv`.

## Question 6 — XGBoost Learning Rate Comparison

**Answer: `eta = 0.1`**

### Objective

Compare two XGBoost regression models using different learning rates to identify the configuration that achieves the lowest validation RMSE.

### Model Configuration

Both models were trained using the same hyperparameters, except for the learning rate (`eta`).

| Parameter        | Model A          | Model B          |
| ---------------- | ---------------- | ---------------- |
| eta              | 0.3              | 0.1              |
| max_depth        | 6                | 6                |
| min_child_weight | 1                | 1                |
| objective        | reg:squarederror | reg:squarederror |
| nthread          | 8                | 8                |
| seed             | 1                | 1                |
| num_boost_round  | 100              | 100              |

### Evaluation Results

| Learning Rate | Training RMSE | Validation RMSE |
| ------------- | ------------: | --------------: |
| 0.3           |      0.901056 |        1.826458 |
| **0.1**       |  **1.311918** |    **1.724810** |

### Key Findings

- The model with `eta=0.1` achieved the lowest validation RMSE of **1.724810**.
- The model with `eta=0.3` achieved a lower training RMSE but a higher validation RMSE.
- Reducing the learning rate from 0.3 to 0.1 improved validation RMSE by approximately **5.57%**.
- The results suggest that a smaller learning rate provided better generalization for this dataset under the tested configuration.

### Conclusion

The best learning rate was **`eta = 0.1`**, achieving a validation RMSE of **1.724810 MPG** after 100 boosting rounds.

The complete comparison is available in `question6_results.csv`.
