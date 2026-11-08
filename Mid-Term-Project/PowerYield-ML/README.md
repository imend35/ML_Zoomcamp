# PowerYield ML ⚡

### Power Plant Energy Output Prediction with Machine Learning

PowerYield ML is an end-to-end machine learning project that predicts the net electrical power output of a Combined Cycle Power Plant using environmental and operational measurements.

Developed as part of the **DataTalksClub Machine Learning Zoomcamp 2026 — Midterm Project**.

## 1. Problem Description

Accurate power output prediction can support energy production planning, operational analysis, and decision-making.

This project predicts net electrical power output (**PE**, measured in MW) using four input features:

| Feature | Description              |
| ------- | ------------------------ |
| AT      | Ambient Temperature (°C) |
| V       | Exhaust Vacuum (cm Hg)   |
| AP      | Ambient Pressure (mbar)  |
| RH      | Relative Humidity (%)    |

**Target:** PE — Net Electrical Power Output (MW).

## 2. Dataset

**Source:** UCI Machine Learning Repository — Combined Cycle Power Plant dataset.

- Original observations: 9,568
- Input features: 4
- Target variable: PE
- Missing values: 0
- Duplicate rows: 41
- Observations after duplicate removal: 9,527

The dataset is available in `data/power_plant.csv`.

## 3. Exploratory Data Analysis

EDA was performed in `notebooks/01_eda.ipynb`.

Key observations:

- Ambient Temperature (AT) has a strong negative correlation with power output.
- Exhaust Vacuum (V) also has a strong negative correlation with power output.
- Ambient Pressure (AP) and Relative Humidity (RH) show positive correlations with power output.
- No missing values were detected.

## 4. Model Training and Evaluation

The following regression algorithms were evaluated:

| Model             | Validation RMSE (MW) | Validation R² |
| ----------------- | -------------------: | ------------: |
| Linear Regression |               4.5085 |        0.9280 |
| Random Forest     |               3.5114 |        0.9563 |
| XGBoost           |               3.4050 |        0.9589 |
| Tuned XGBoost     |           **3.1599** |    **0.9646** |

Hyperparameter optimization was performed using `RandomizedSearchCV`.

**Final model:** Tuned XGBoost Regressor.

Final model performance reported by `train.py`:

| Metric |   Test Result |
| ------ | ------------: |
| MAE    |     2.2678 MW |
| RMSE   | **3.2514 MW** |
| R²     |        0.9640 |

The trained model is saved as `models/poweryield_model.json`.

## 5. Project Structure

```text
PowerYield-ML/
├── data/
│   └── power_plant.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb
├── models/
│   ├── poweryield_model.json
│   └── metrics.json
├── tests/
│   └── test_api.py
├── train.py
├── predict.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-prod.txt
└── README.md
```

## 6. Local Setup

Use Python 3.11.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 7. Train the Model

From the project directory:

```bash
python train.py
```

The training script creates:

- `models/poweryield_model.json`
- `models/metrics.json`

## 8. Run the FastAPI Application

Start the API:

```bash
python -m uvicorn predict:app --host 127.0.0.1 --port 9696
```

Swagger documentation:

`http://127.0.0.1:9696/docs`

Available endpoints:

- `GET /` — API information
- `GET /health` — Health check
- `POST /predict` — Power output prediction

Example prediction request:

```json
{
  "AT": 25.18,
  "V": 62.96,
  "AP": 1020.04,
  "RH": 59.08
}
```

Example response:

```json
{
  "predicted_power_output_mw": 444.9519,
  "unit": "MW"
}
```

## 9. Run with Docker

Build the Docker image:

```bash
docker build -t poweryield-ml:1.0 .
```

Run the container:

```bash
docker run --rm -p 9697:9696 --name poweryield-api poweryield-ml:1.0
```

Open the Docker-hosted Swagger interface:

`http://127.0.0.1:9697/docs`

Health check:

```bash
curl http://127.0.0.1:9697/health
```

## 10. Automated Testing

Run the API tests:

```bash
python -m pytest tests/ -v
```

**Latest verified local test result: 4 passed.**

Tests cover:

- Root endpoint
- Health endpoint
- Successful prediction
- Invalid input validation

## 11. Technology Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, FastAPI, Pydantic, Pytest, Docker, Jupyter Notebook.

## 12. Live Cloud Deployment

PowerYield ML is deployed on Render using Docker.

- **Platform:** Render
- **Runtime:** Docker / Python 3.11
- **Model:** Tuned XGBoost Regressor
- **API Framework:** FastAPI
- **Live API:** https://poweryield-ml-api.onrender.com
- **Interactive API Documentation:** https://poweryield-ml-api.onrender.com/docs
- **Health Check:** https://poweryield-ml-api.onrender.com/health

### Example Prediction

Request:

```json
{
  "AT": 25.18,
  "V": 62.96,
  "AP": 1020.04,
  "RH": 59.08
}
```

Response:

```json
{
  "predicted_power_output_mw": 444.9519,
  "unit": "MW"
}
```

The API successfully returned HTTP 200 during live deployment testing.

**Note:** The free Render instance may spin down after inactivity, so the first request can take longer to complete.

## Interactive Web Application

PowerYield ML includes an interactive, responsive web interface that allows users to predict the net electrical power output of a Combined Cycle Power Plant using a trained XGBoost regression model.

### Live Demo

**Web Application:** https://poweryield-ml-api.onrender.com/

**Swagger API Documentation:** https://poweryield-ml-api.onrender.com/docs

**Health Check:** https://poweryield-ml-api.onrender.com/health

### How to Use

1. Open the live web application.
2. Enter the four operating parameters:
   - **AT:** Ambient Temperature (°C)
   - **V:** Exhaust Vacuum (cm Hg)
   - **AP:** Ambient Pressure (mbar)
   - **RH:** Relative Humidity (%)
3. Click **Predict Power Output**.
4. View the predicted electrical power output in megawatts (MW).

### Example Prediction

| Feature                    |           Value |
| -------------------------- | --------------: |
| AT                         |           25.18 |
| V                          |           62.96 |
| AP                         |         1020.04 |
| RH                         |           59.08 |
| **Predicted Power Output** | **444.9519 MW** |

### Application Architecture

The application integrates:

- **Frontend:** HTML, CSS, and JavaScript
- **Backend:** FastAPI
- **Machine Learning:** XGBoost Regression
- **Containerization:** Docker
- **Continuous Integration:** GitHub Actions
- **Cloud Deployment:** Render

The frontend sends user inputs to the `/predict` API endpoint and displays the model's prediction without requiring users to interact directly with the API.

### Model Performance

The final model achieved the following performance on the held-out test dataset:

| Metric |     Value |
| ------ | --------: |
| MAE    | 2.2678 MW |
| RMSE   | 3.2514 MW |
| R²     |    0.9640 |

### Deployment Notes

The application is deployed using Render's free service tier. After periods of inactivity, the service may take additional time to respond while the instance starts.

The application is intended for educational and demonstration purposes.

## 13. Author

**Esila Nur Demirci**

Senior Business Analyst | AI Engineer | Data Engineer | Data Analyst

Developed for the DataTalksClub Machine Learning Zoomcamp 2026.
