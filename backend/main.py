"""
Run locally:
    cd backend
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import EmployeeInput, FactorContribution, PredictionResponse

ARTIFACTS_DIR = Path(__file__).resolve().parent / "model_artifacts"

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Serves attrition-risk predictions from a Logistic Regression "
                 "model trained on the IBM HR Analytics dataset.",
    version="1.0.0",
)

# Allow the Streamlit frontend (a different origin/port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = joblib.load(ARTIFACTS_DIR / "pipeline.pkl")

with open(ARTIFACTS_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS: list[str] = json.load(f)

with open(ARTIFACTS_DIR / "threshold.json") as f:
    THRESHOLD_INFO: dict = json.load(f)
    OPTIMAL_THRESHOLD: float = THRESHOLD_INFO["optimal_threshold"]

with open(ARTIFACTS_DIR / "preprocessing_reference.json") as f:
    PREP_REF: dict = json.load(f)

BINARY_ENCODERS = PREP_REF["binary_encoders"]          # e.g. {"OverTime": ["No","Yes"]}
MULTI_COL_OPTIONS = PREP_REF["multi_col_options"]       # dropdown choices per categorical col
SATISFACTION_COLS = PREP_REF["satisfaction_cols"]
JOB_LEVEL_MEDIAN_INCOME = {int(k): v for k, v in PREP_REF["job_level_median_income"].items()}
FEATURE_COEFFICIENTS = PREP_REF["feature_coefficients"]  # for explaining predictions


def build_feature_row(employee: EmployeeInput) -> pd.DataFrame:

    row = employee.model_dump()

    # --- Step 1: binary encode (must match LabelEncoder's fit order) ------
    for col in ["OverTime", "Gender"]:
        classes = BINARY_ENCODERS[col]          # e.g. ["No", "Yes"] -> No=0, Yes=1
        row[col] = classes.index(row[col])

    multi_cols = list(MULTI_COL_OPTIONS.keys())
    df_row = pd.DataFrame([row])

    for col in multi_cols:
        df_row[col] = pd.Categorical(df_row[col], categories=MULTI_COL_OPTIONS[col])

    df_row = pd.get_dummies(df_row, columns=multi_cols, drop_first=True)

    df_row["Satisfaction_Score"] = df_row[SATISFACTION_COLS].mean(axis=1)
    df_row["Tenure_Ratio"] = df_row["YearsInCurrentRole"] / (df_row["TotalWorkingYears"] + 1)
    role_median = JOB_LEVEL_MEDIAN_INCOME.get(int(employee.JobLevel))
    df_row["Income_vs_Role_Median"] = df_row["MonthlyIncome"] - role_median

    df_row = df_row.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df_row.values


def explain_prediction(df_row: pd.DataFrame, top_n: int = 5):

    scaler = pipeline.named_steps["scaler"]
    scaled_values = scaler.transform(df_row)[0]

    contributions = [
        (feature, scaled_values[i] * FEATURE_COEFFICIENTS[feature])
        for i, feature in enumerate(FEATURE_COLUMNS)
    ]
    contributions.sort(key=lambda x: x[1], reverse=True)

    top_risk = [FactorContribution(feature=f, contribution=round(c, 4))
                for f, c in contributions[:top_n] if c > 0]
    top_protective = [FactorContribution(feature=f, contribution=round(c, 4))
                       for f, c in contributions[-top_n:] if c < 0]
    top_protective.sort(key=lambda x: x.contribution)  # most protective first

    return top_risk, top_protective


@app.get("/health")
def health_check():
    """Simple endpoint to confirm the API is up and the model is loaded."""
    return {"status": "ok", "model_loaded": pipeline is not None,
            "optimal_threshold": OPTIMAL_THRESHOLD}


@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeInput):
    try:
        df_row = build_feature_row(employee)
        probability = float(pipeline.predict_proba(df_row)[0, 1])
        top_risk, top_protective = explain_prediction(df_row)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    if probability >= 0.60:
        tier = "High Risk"
    elif probability >= 0.35:
        tier = "Medium Risk"
    else:
        tier = "Low Risk"

    return PredictionResponse(
        attrition_probability=round(probability, 4),
        risk_tier=tier,
        decision_threshold_used=OPTIMAL_THRESHOLD,
        predicted_attrition=probability >= OPTIMAL_THRESHOLD,
        top_risk_factors=top_risk,
        top_protective_factors=top_protective,
    )