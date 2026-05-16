import os
import time

import pandas as pd
from axiom_py import Client
from dotenv import load_dotenv
from litestar import Litestar, Request, get, post
from pydantic import BaseModel

from app.logger_setup import setup_logging
from app.model_utils import FEATURE_COLS, predict_churn, predict_proba, transformer

load_dotenv()

logger = setup_logging()

axiom_client = Client(os.getenv("AXIOM_TOKEN"))
AXIOM_DATASET = os.getenv("AXIOM_DATASET")


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


@get("/")
async def root(request: Request) -> dict:
    logger.info("Root endpoint called")
    axiom_client.ingest_events(
        dataset=AXIOM_DATASET,
        events=[
            {
                "endpoint": "/",
                "method": "GET",
                "status_code": 200,
                "user_agent": request.headers.get("user-agent", "unknown"),
            }
        ],
    )
    return {"message": "Welcome to the Churn Prediction API"}


@get("/health")
async def health(request: Request) -> dict:
    logger.info("Health check called")
    axiom_client.ingest_events(
        dataset=AXIOM_DATASET,
        events=[
            {
                "endpoint": "/health",
                "method": "GET",
                "status_code": 200,
                "user_agent": request.headers.get("user-agent", "unknown"),
            }
        ],
    )
    return {"status": "healthy"}


@post("/predict")
async def predict(data: ChurnRequest, request: Request) -> dict:
    start_time = time.time()

    raw = pd.DataFrame([data.model_dump()])[FEATURE_COLS]
    features = transformer.transform(raw)[0].tolist()
    result = predict_churn(features)
    probability = predict_proba(features)

    response_time_ms = (time.time() - start_time) * 1000

    logger.info(f"Prediction requested | input: {data.model_dump()} | result: {result}")

    axiom_client.ingest_events(
        dataset=AXIOM_DATASET,
        events=[
            {
                "endpoint": "/predict",
                "method": "POST",
                "status_code": 201,
                "response_time_ms": response_time_ms,
                "credit_score": data.CreditScore,
                "age": data.Age,
                "balance": data.Balance,
                "geography": data.Geography,
                "gender": data.Gender,
                "num_of_products": data.NumOfProducts,
                "is_active_member": data.IsActiveMember,
                "predicted_class": result,
                "predicted_label": "churn" if result == 1 else "no_churn",
                "churn_probability": probability,
                "user_agent": request.headers.get("user-agent", "unknown"),
                "content_type": request.headers.get("content-type", "unknown"),
            }
        ],
    )

    return {"churn_prediction": result, "churn_probability": probability}


app = Litestar(route_handlers=[root, health, predict])
