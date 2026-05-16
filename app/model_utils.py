import joblib
import pandas as pd

model = joblib.load("data/model.pkl")
transformer = joblib.load("column_transformer/column_transformer.joblib")

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


def predict_churn(features: list[float]) -> int:
    prediction = model.predict([features])
    return int(prediction[0])


def predict_proba(features: list[float]) -> float:
    proba = model.predict_proba([features])
    return round(float(proba[0][1]), 4)


if __name__ == "__main__":
    df = pd.read_csv("data/Churn_Modelling.csv", nrows=1)
    raw_row = df.iloc[0:1][FEATURE_COLS]
    print(f"Raw input:\n{raw_row.to_string()}\n")

    features = transformer.transform(raw_row)[0].tolist()
    print(f"Transformed features: {features}\n")
    print(f"Prediction:           {predict_churn(features)}")
