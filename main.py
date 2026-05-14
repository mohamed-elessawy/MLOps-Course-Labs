import pandas as pd
from litestar import Litestar, get, post
from pydantic import BaseModel

from app.logger_setup import setup_logging
from app.model_utils import FEATURE_COLS, predict_churn, transformer


logger = setup_logging()


class ChurnRequest(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@get("/")
async def root() -> dict:
    logger.info("Root endpoint called")
    return {"message": "Welcome to the Churn Prediction API"}


@get("/health")
async def health() -> dict:
    logger.info("Health check called")
    return {"status": "healthy"}


@post("/predict")
async def predict(data: ChurnRequest) -> dict:

    raw = pd.DataFrame([data.model_dump()])[FEATURE_COLS]
    features = transformer.transform(raw)[0].tolist()

    result = predict_churn(features)
    logger.info(f"Prediction requested | input: {data.model_dump()} | result: {result}")

    return {"churn_prediction": result}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Litestar(
    route_handlers=[root, health, predict],
)
