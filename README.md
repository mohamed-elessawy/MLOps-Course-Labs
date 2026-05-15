# MLOps Course Labs

Welcome to the lab repository for the [MLOps Course](https://github.com/Heba-Atef99/MLOps-Course).

Throughout this hands-on journey, you'll develop a **Bank Customer Churn Prediction** application—starting from the research phase and progressing through the full MLOps lifecycle, all the way to deployment.

## Churn Prediction API

A Litestar API serving the churn prediction model with logging and tests.

### Setup

```bash
uv sync
uv run pre-commit install
# place your model in data/model.joblib
uv run litestar --app main:app run --reload
```

Swagger UI: http://localhost:8000/schema/swagger

### Tests

```bash
uv run pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
```

### Endpoints

| Method | Path       | Description              |
| ------ | ---------- | ------------------------ |
| GET    | `/`        | Welcome message          |
| GET    | `/health`  | Health check             |
| POST   | `/predict` | Returns churn prediction |

# test