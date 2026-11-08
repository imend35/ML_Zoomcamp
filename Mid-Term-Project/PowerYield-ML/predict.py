
from pathlib import Path
from fastapi.responses import FileResponse
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "poweryield_model.json"

app = FastAPI(
    title="PowerYield ML API",
    description="Combined Cycle Power Plant Output Prediction",
    version="1.0.0"
)

model = XGBRegressor()
model.load_model(str(MODEL_PATH))


class PowerPlantInput(BaseModel):
    AT: float = Field(..., ge=0, le=50)
    V: float = Field(..., ge=0, le=100)
    AP: float = Field(..., ge=900, le=1100)
    RH: float = Field(..., ge=0, le=100)


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(
        BASE_DIR / "templates" / "index.html",
        media_type="text/html"
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: PowerPlantInput):
    input_df = pd.DataFrame(
        [[data.AT, data.V, data.AP, data.RH]],
        columns=["AT", "V", "AP", "RH"]
    )

    prediction = float(model.predict(input_df)[0])

    return {
        "predicted_power_output_mw": round(prediction, 4),
        "unit": "MW"
    }
