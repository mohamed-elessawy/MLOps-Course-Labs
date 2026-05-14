from litestar.testing import TestClient

from app.model_utils import predict_churn, transformer
from main import app
import pandas as pd
import pytest

# sample test
SAMPLE_ROW = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
}

FEATURE_COLS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def test_predict_churn_returns_valid_prediction():
    raw = pd.DataFrame([SAMPLE_ROW])[FEATURE_COLS]
    features = transformer.transform(raw)[0].tolist()
    result = predict_churn(features)
    assert result in (0, 1)


def test_predict_churn_with_garbage_input():
    with pytest.raises(Exception):
        garbage_row = {col: "abc" for col in FEATURE_COLS}
        raw = pd.DataFrame([garbage_row])[FEATURE_COLS]
        features = transformer.transform(raw)[0].tolist()
        predict_churn(features)


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------


def test_predict_endpoint():
    with TestClient(app=app) as client:
        response = client.post("/predict", json=SAMPLE_ROW)
        assert response.status_code == 201
        body = response.json()
        assert "churn_prediction" in body
        assert body["churn_prediction"] in (0, 1)


def test_health_endpoint():
    with TestClient(app=app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_root_endpoint():
    with TestClient(app=app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


def test_predict_invalid_input():
    with TestClient(app=app) as client:
        response = client.post("/predict", json={"invalid_field": "garbage"})
        assert response.status_code == 400
