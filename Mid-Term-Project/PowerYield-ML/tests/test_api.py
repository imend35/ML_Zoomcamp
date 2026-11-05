
from fastapi.testclient import TestClient
from predict import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    payload = {
        "AT": 25.18,
        "V": 62.96,
        "AP": 1020.04,
        "RH": 59.08,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    result = response.json()

    assert "predicted_power_output_mw" in result
    assert result["unit"] == "MW"
    assert 400 < result["predicted_power_output_mw"] < 510


def test_invalid_input():
    payload = {
        "AT": -100,
        "V": 62.96,
        "AP": 1020.04,
        "RH": 59.08,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
