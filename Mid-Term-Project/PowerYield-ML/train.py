
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "power_plant.csv"
MODEL_DIR = BASE_DIR / "models"

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"

BEST_PARAMS = {
    "subsample": 0.9,
    "reg_lambda": 1,
    "n_estimators": 700,
    "min_child_weight": 5,
    "max_depth": 6,
    "learning_rate": 0.05,
    "colsample_bytree": 0.9,
}


def main():
    print("PowerYield ML — Training Pipeline")

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df[FEATURES]
    y = df[TARGET]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = XGBRegressor(
        **BEST_PARAMS,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_train_val, y_train_val)
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "training_samples": len(X_train_val),
        "test_samples": len(X_test),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "poweryield_model.json"
    model.save_model(str(model_path))

    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nFinal Test Results")
    print(f"MAE:  {metrics['mae']:.4f} MW")
    print(f"RMSE: {metrics['rmse']:.4f} MW")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"\nModel saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
