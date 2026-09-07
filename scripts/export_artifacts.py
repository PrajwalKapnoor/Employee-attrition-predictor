"""
export_artifacts.py
====================
This is the ONE bridge file between your Jupyter notebook and the deployed app.

It re-runs the exact same steps as `employee_attrition_predictor.ipynb`
(load -> preprocess -> feature-engineer -> train -> optimise threshold ->
risk-tier) and then SAVES the results in two forms:

  1. Model artifacts (backend/model_artifacts/) -> used by FastAPI to serve
     live predictions.
  2. Pre-computed chart data (frontend/assets/) -> used by Streamlit to
     render the EDA and Business Metrics pages WITHOUT re-running any of
     this logic. The frontend just loads JSON and draws it.

Run this once (and again any time the notebook logic changes):
    python scripts/export_artifacts.py

Why this design?
  - The backend should only know how to do ONE thing: turn a single
    employee's data into a probability. It should not know how to draw
    charts or hold the full training dataset in memory.
  - The frontend should only know how to display things. It should not
    re-train a model or re-compute statistics every time someone visits
    the EDA page.
  - This script is the only place both worlds meet.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# PATHS — everything is relative to the project root so this script works
# no matter where you run it from, as long as the folder structure is intact.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
BACKEND_ARTIFACTS_DIR = ROOT / "backend" / "model_artifacts"
FRONTEND_ASSETS_DIR = ROOT / "frontend" / "assets"

BACKEND_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Plotly "dark" template so every chart already matches the app's black theme
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_LEAVE = "#ef5350"   # red   — same meaning as in your notebook
COLOR_STAY = "#42a5f5"    # blue
COLOR_HIGH = "#ef5350"
COLOR_MED = "#ffca28"     # amber
COLOR_LOW = "#66bb6a"     # green


def fig_to_dict(fig: go.Figure) -> dict:
    """Convert a Plotly figure to a plain dict so it can be saved as JSON
    and later reloaded in Streamlit with `go.Figure(saved_dict)`."""
    return json.loads(fig.to_json())


# ===========================================================================
# STEP 1 — LOAD & VALIDATE (same as notebook Cell 8-13)
# ===========================================================================
print("[1/7] Loading dataset...")
df = pd.read_csv(DATA_PATH)
raw_shape = df.shape
missing_total = int(df.isnull().sum().sum())

CONSTANT_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
# For genuinely constant columns, show the single value. EmployeeNumber isn't
# constant (it's an ID) so it gets a summary instead of all 1,470 raw values.
constant_col_values = {}
for c in CONSTANT_COLS:
    n_unique = df[c].nunique()
    if n_unique == 1:
        constant_col_values[c] = {"type": "constant", "value": df[c].unique().tolist()[0]}
    else:
        constant_col_values[c] = {"type": "unique_id", "n_unique": int(n_unique)}
df.drop(columns=CONSTANT_COLS, inplace=True)

attrition_counts = df["Attrition"].value_counts().to_dict()
attrition_rate = (df["Attrition"] == "Yes").mean() * 100

df["Attrition_Bin"] = (df["Attrition"] == "Yes").astype(int)

# ===========================================================================
# STEP 2 — EDA CHARTS (Act 1 + Act 2 from the notebook, rebuilt in Plotly)
# ===========================================================================
print("[2/7] Building EDA charts...")
eda_figures = {}

# --- Act 1, Panel 1: Attrition rate by Department -------------------------
dept_rate = (
    df.groupby("Department")["Attrition_Bin"].mean().sort_values() * 100
).reset_index()
dept_rate.columns = ["Department", "AttritionRate"]
dept_rate["AboveAvg"] = dept_rate["AttritionRate"] > attrition_rate
fig = px.bar(
    dept_rate, x="AttritionRate", y="Department", orientation="h",
    color="AboveAvg", color_discrete_map={True: COLOR_LEAVE, False: COLOR_STAY},
    title="Attrition Rate by Department (red = above company average)",
    labels={"AttritionRate": "Attrition Rate (%)"}, template=PLOTLY_TEMPLATE,
)
fig.add_vline(x=attrition_rate, line_dash="dash", line_color=COLOR_MED,
              annotation_text=f"Company avg {attrition_rate:.1f}%")
fig.update_layout(showlegend=False)
eda_figures["attrition_by_department"] = fig_to_dict(fig)

# --- Act 1, Panel 2: Attrition rate by Job Role ----------------------------
role_rate = (
    df.groupby("JobRole")["Attrition_Bin"].mean().sort_values() * 100
).reset_index()
role_rate.columns = ["JobRole", "AttritionRate"]
role_rate["AboveAvg"] = role_rate["AttritionRate"] > attrition_rate
fig = px.bar(
    role_rate, x="AttritionRate", y="JobRole", orientation="h",
    color="AboveAvg", color_discrete_map={True: COLOR_LEAVE, False: COLOR_STAY},
    title="Attrition Rate by Job Role (highest-risk roles in red)",
    labels={"AttritionRate": "Attrition Rate (%)"}, template=PLOTLY_TEMPLATE,
)
fig.add_vline(x=attrition_rate, line_dash="dash", line_color=COLOR_MED,
              annotation_text=f"Company avg {attrition_rate:.1f}%")
fig.update_layout(showlegend=False)
eda_figures["attrition_by_role"] = fig_to_dict(fig)

# --- Act 1, Panel 3: OverTime vs Attrition (strongest single raw signal) --
ot = df.groupby(["OverTime", "Attrition"]).size().unstack(fill_value=0)
ot_pct = (ot.div(ot.sum(axis=1), axis=0) * 100).reset_index()
ot_long = ot_pct.melt(id_vars="OverTime", value_vars=["No", "Yes"],
                       var_name="Attrition", value_name="Percent")
fig = px.bar(
    ot_long, x="OverTime", y="Percent", color="Attrition", barmode="group",
    color_discrete_map={"No": COLOR_STAY, "Yes": COLOR_LEAVE},
    title="OverTime vs Attrition — the #1 single predictor",
    labels={"Percent": "% of Employees"}, template=PLOTLY_TEMPLATE,
)
eda_figures["overtime_vs_attrition"] = fig_to_dict(fig)

# --- Act 2, Panel 1: Monthly Income distribution, leavers vs stayers ------
fig = px.histogram(
    df, x="MonthlyIncome", color="Attrition", barmode="overlay", nbins=40,
    color_discrete_map={"No": COLOR_STAY, "Yes": COLOR_LEAVE}, opacity=0.6,
    title="Monthly Income: Leavers vs Stayers (leavers skew lower income)",
    template=PLOTLY_TEMPLATE,
)
eda_figures["income_distribution"] = fig_to_dict(fig)

# --- Act 2, Panel 2: Years Since Last Promotion by attrition status -------
fig = px.box(
    df, x="Attrition", y="YearsSinceLastPromotion", color="Attrition",
    color_discrete_map={"No": COLOR_STAY, "Yes": COLOR_LEAVE},
    title="Years Since Last Promotion by Attrition Status",
    template=PLOTLY_TEMPLATE,
)
eda_figures["promotion_gap"] = fig_to_dict(fig)

# --- Act 2, Panel 3: Top numeric correlations with attrition --------------
numeric_df = df.select_dtypes(include=[np.number])
corr_target = numeric_df.corr()["Attrition_Bin"].drop("Attrition_Bin").sort_values()
top_corr = pd.concat([corr_target.head(7), corr_target.tail(7)]).reset_index()
top_corr.columns = ["Feature", "Correlation"]
top_corr["Direction"] = np.where(top_corr["Correlation"] < 0, "Leaves less", "Leaves more")
fig = px.bar(
    top_corr, x="Correlation", y="Feature", orientation="h", color="Direction",
    color_discrete_map={"Leaves less": COLOR_LEAVE, "Leaves more": COLOR_LOW},
    title="Top Feature Correlations with Attrition",
    template=PLOTLY_TEMPLATE,
)
eda_figures["top_correlations"] = fig_to_dict(fig)

# --- Target distribution (class imbalance) --------------------------------
fig = px.pie(
    names=list(attrition_counts.keys()), values=list(attrition_counts.values()),
    color=list(attrition_counts.keys()),
    color_discrete_map={"No": COLOR_STAY, "Yes": COLOR_LEAVE},
    title=f"Target Distribution — {attrition_rate:.1f}% Attrition Rate (class imbalance ~5:1)",
    template=PLOTLY_TEMPLATE, hole=0.45,
)
eda_figures["target_distribution"] = fig_to_dict(fig)

# ===========================================================================
# STEP 3 — PREPROCESSING (same as notebook Cell 20-26)
# ===========================================================================
print("[3/7] Preprocessing...")
df_pre = df.copy()
df_pre["Attrition"] = (df_pre["Attrition"] == "Yes").astype(int)

binary_cols = ["OverTime", "Gender"]
binary_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df_pre[col] = le.fit_transform(df_pre[col])
    binary_encoders[col] = le.classes_.tolist()  # e.g. ['No', 'Yes'] -> index 0/1

multi_cols = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]
multi_col_options = {c: sorted(df[c].unique().tolist()) for c in multi_cols}

df_pre = pd.get_dummies(df_pre, columns=multi_cols, drop_first=True)
df_pre.drop(columns=["Attrition_Bin"], inplace=True, errors="ignore")

# ===========================================================================
# STEP 4 — FEATURE ENGINEERING (same as notebook Cell 28-32)
# ===========================================================================
print("[4/7] Feature engineering...")
df_feat = df_pre.copy()

satisfaction_cols = ["EnvironmentSatisfaction", "JobSatisfaction",
                      "RelationshipSatisfaction", "WorkLifeBalance"]
df_feat["Satisfaction_Score"] = df_feat[satisfaction_cols].mean(axis=1)

df_feat["Tenure_Ratio"] = (
    df_feat["YearsInCurrentRole"] / (df_feat["TotalWorkingYears"] + 1)
)

role_median_income = df_feat.groupby("JobLevel")["MonthlyIncome"].transform("median")
df_feat["Income_vs_Role_Median"] = df_feat["MonthlyIncome"] - role_median_income
# Needed at inference time to reproduce this feature for a single new employee
job_level_median_income = (
    df_feat.groupby("JobLevel")["MonthlyIncome"].median().to_dict()
)

# ===========================================================================
# STEP 5 — MODELLING (same as notebook Cell 34-38)
# ===========================================================================
print("[5/7] Training model with 5-fold stratified CV...")
TARGET = "Attrition"
FEATURES = [c for c in df_feat.columns if c != TARGET]

X = df_feat[FEATURES].values
y = df_feat[TARGET].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42, C=0.5
    )),
])

scoring = {"roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1"}
cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
cv_summary = {
    metric: {"mean": float(cv_results[f"test_{metric}"].mean()),
             "std": float(cv_results[f"test_{metric}"].std())}
    for metric in scoring
}

# Refit on the full dataset — this is the pipeline that gets deployed
model.fit(X, y)
y_proba = model.predict_proba(X)[:, 1]

# ===========================================================================
# STEP 6 — THRESHOLD OPTIMISATION & BUSINESS COST (same as Cell 40-42)
# ===========================================================================
print("[6/7] Optimising decision threshold on business cost...")
COST_FALSE_NEGATIVE = 6.0
COST_FALSE_POSITIVE = 0.5
avg_monthly_income = float(df["MonthlyIncome"].mean())

thresholds = np.arange(0.10, 0.91, 0.01)
business_costs = []
for thresh in thresholds:
    y_pred_t = (y_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred_t).ravel()
    cost = (fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE) * avg_monthly_income
    business_costs.append(cost)

best_idx = int(np.argmin(business_costs))
best_thresh = float(thresholds[best_idx])
best_cost = float(business_costs[best_idx])
default_idx = int(np.argmin(np.abs(thresholds - 0.50)))
default_cost = float(business_costs[default_idx])

y_pred_optimal = (y_proba >= best_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y, y_pred_optimal).ravel()
auc_score = float(roc_auc_score(y, y_proba))

# --- Business metrics charts ----------------------------------------------
metrics_figures = {}

fpr, tpr, _ = roc_curve(y, y_proba)
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Logistic Regression (AUC={auc_score:.3f})",
                          line=dict(color=COLOR_STAY, width=3)))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random classifier",
                          line=dict(color="#555", width=1, dash="dash")))
fig.update_layout(title="ROC Curve — Discrimination Power", xaxis_title="False Positive Rate",
                   yaxis_title="True Positive Rate (Recall)", template=PLOTLY_TEMPLATE)
metrics_figures["roc_curve"] = fig_to_dict(fig)

prec, rec, _ = precision_recall_curve(y, y_proba)
fig = go.Figure()
fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="Precision-Recall",
                          line=dict(color="#ab47bc", width=3)))
fig.add_hline(y=float(y.mean()), line_dash="dash", line_color=COLOR_LEAVE,
              annotation_text=f"Baseline (random): {y.mean():.2f}")
fig.update_layout(title="Precision-Recall Curve (better than ROC for imbalanced classes)",
                   xaxis_title="Recall (% of leavers caught)",
                   yaxis_title="Precision (% of flagged who actually leave)",
                   template=PLOTLY_TEMPLATE)
metrics_figures["precision_recall_curve"] = fig_to_dict(fig)

cm_labels = ["Stays", "Leaves"]
cm_matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]
fig = px.imshow(cm_matrix, x=cm_labels, y=cm_labels, text_auto=True,
                 color_continuous_scale="Blues",
                 labels=dict(x="Predicted", y="Actual", color="Count"),
                 title=f"Confusion Matrix (threshold={best_thresh:.2f})")
fig.update_layout(template=PLOTLY_TEMPLATE)
metrics_figures["confusion_matrix"] = fig_to_dict(fig)

fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=[c / 1e6 for c in business_costs], mode="lines",
                          line=dict(color=COLOR_MED, width=3), fill="tozeroy"))
fig.add_vline(x=best_thresh, line_dash="dash", line_color=COLOR_LEAVE,
              annotation_text=f"Optimal {best_thresh:.2f}")
fig.add_vline(x=0.50, line_dash="dot", line_color=COLOR_STAY, annotation_text="Default 0.50")
fig.update_layout(title="Business Cost Curve vs Decision Threshold<br>"
                         "(Total cost = FN×6mo salary + FP×0.5mo salary)",
                   xaxis_title="Decision Threshold", yaxis_title="Total Business Cost ($M)",
                   template=PLOTLY_TEMPLATE)
metrics_figures["business_cost_curve"] = fig_to_dict(fig)

# Feature importance (top 20 by absolute coefficient)
lr_model = model.named_steps["clf"]
coef_df = pd.DataFrame({"feature": FEATURES, "coef": lr_model.coef_[0]})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=True).tail(20)
coef_df["direction"] = np.where(coef_df["coef"] > 0, "Increases risk", "Decreases risk")
fig = px.bar(coef_df, x="coef", y="feature", orientation="h", color="direction",
             color_discrete_map={"Increases risk": COLOR_LOW, "Decreases risk": COLOR_LEAVE},
             title="Top 20 Feature Coefficients — What Drives Attrition Prediction?",
             labels={"coef": "Logistic Regression Coefficient (log-odds)"},
             template=PLOTLY_TEMPLATE)
metrics_figures["feature_importance"] = fig_to_dict(fig)
# Save the raw table too — the prediction page reuses these coefficients
# to explain individual predictions (see backend/main.py)
feature_coefficients = dict(zip(FEATURES, lr_model.coef_[0].tolist()))

# ===========================================================================
# STEP 7 — RISK TIER TABLE (same as notebook Cell 45-48)
# ===========================================================================
print("[7/7] Building risk tier table...")
risk_df = pd.read_csv(DATA_PATH).drop(columns=CONSTANT_COLS, errors="ignore")
risk_df["Attrition_Probability"] = y_proba
risk_df["Actual_Attrition"] = (risk_df["Attrition"] == "Yes").astype(int)
risk_df["Replacement_Cost_Est"] = risk_df["MonthlyIncome"] * COST_FALSE_NEGATIVE


def assign_risk_tier(p):
    if p >= 0.60:
        return "High Risk"
    elif p >= 0.35:
        return "Medium Risk"
    return "Low Risk"


risk_df["Risk_Tier"] = risk_df["Attrition_Probability"].apply(assign_risk_tier)

tier_summary = (
    risk_df.groupby("Risk_Tier")
    .agg(Employees=("JobRole", "count"),
         Avg_Attrition_Prob=("Attrition_Probability", "mean"),
         Actual_Leavers=("Actual_Attrition", "sum"),
         Avg_Replacement_Cost=("Replacement_Cost_Est", "mean"),
         Total_At_Stake=("Replacement_Cost_Est", "sum"))
    .round(2)
    .reset_index()
    .to_dict(orient="records")
)

top_risk = (
    risk_df[["JobRole", "Department", "MonthlyIncome", "YearsAtCompany",
             "OverTime", "Attrition_Probability", "Risk_Tier", "Replacement_Cost_Est"]]
    .sort_values("Attrition_Probability", ascending=False)
    .head(10)
    .round(3)
    .to_dict(orient="records")
)

# Risk tier distribution + cost-at-stake charts
tier_colors = {"High Risk": COLOR_HIGH, "Medium Risk": COLOR_MED, "Low Risk": COLOR_LOW}
fig = px.histogram(risk_df, x="Attrition_Probability", color="Risk_Tier", nbins=30,
                    color_discrete_map=tier_colors,
                    title="Attrition Probability Distribution by Risk Tier",
                    template=PLOTLY_TEMPLATE)
fig.add_vline(x=best_thresh, line_dash="dash", line_color="white")
metrics_figures["risk_tier_distribution"] = fig_to_dict(fig)

tier_order = ["High Risk", "Medium Risk", "Low Risk"]
tier_cost_map = {t["Risk_Tier"]: t["Total_At_Stake"] for t in tier_summary}
fig = px.bar(x=tier_order, y=[tier_cost_map.get(t, 0) for t in tier_order],
             color=tier_order, color_discrete_map=tier_colors,
             title="Total Replacement Cost At Stake by Risk Tier",
             labels={"x": "Risk Tier", "y": "Total Replacement Cost ($)"},
             template=PLOTLY_TEMPLATE)
metrics_figures["risk_tier_cost"] = fig_to_dict(fig)

# ===========================================================================
# SAVE — BACKEND ARTIFACTS (what FastAPI needs to serve predictions)
# ===========================================================================
print("Saving backend artifacts...")
joblib.dump(model, BACKEND_ARTIFACTS_DIR / "pipeline.pkl")

with open(BACKEND_ARTIFACTS_DIR / "feature_columns.json", "w") as f:
    json.dump(FEATURES, f, indent=2)

with open(BACKEND_ARTIFACTS_DIR / "threshold.json", "w") as f:
    json.dump({
        "optimal_threshold": best_thresh,
        "default_threshold": 0.50,
        "cost_false_negative_months": COST_FALSE_NEGATIVE,
        "cost_false_positive_months": COST_FALSE_POSITIVE,
    }, f, indent=2)

with open(BACKEND_ARTIFACTS_DIR / "preprocessing_reference.json", "w") as f:
    json.dump({
        "constant_cols_dropped": CONSTANT_COLS,
        "binary_cols": binary_cols,
        "binary_encoders": binary_encoders,      # class order for LabelEncoder replay
        "multi_cols": multi_cols,
        "multi_col_options": multi_col_options,  # dropdown choices for the form
        "satisfaction_cols": satisfaction_cols,
        "job_level_median_income": {str(k): v for k, v in job_level_median_income.items()},
        "feature_coefficients": feature_coefficients,
        "model_intercept": float(lr_model.intercept_[0]),
    }, f, indent=2)

# ===========================================================================
# SAVE — FRONTEND ASSETS (what Streamlit needs to render charts, no compute)
# ===========================================================================
print("Saving frontend assets...")
with open(FRONTEND_ASSETS_DIR / "eda_figures.json", "w") as f:
    json.dump(eda_figures, f)

with open(FRONTEND_ASSETS_DIR / "metrics_figures.json", "w") as f:
    json.dump(metrics_figures, f)

with open(FRONTEND_ASSETS_DIR / "dataset_summary.json", "w") as f:
    json.dump({
        "raw_shape": raw_shape,
        "shape_after_cleaning": list(df.drop(columns=["Attrition_Bin"]).shape),
        "missing_values_total": missing_total,
        "constant_cols_dropped": constant_col_values,
        "attrition_counts": {str(k): int(v) for k, v in attrition_counts.items()},
        "attrition_rate_pct": round(float(attrition_rate), 2),
        "cv_metrics": cv_summary,
        "roc_auc_full_fit": round(auc_score, 4),
        "optimal_threshold": round(best_thresh, 2),
        "default_threshold_cost": round(default_cost, 2),
        "optimal_threshold_cost": round(best_cost, 2),
        "cost_savings": round(default_cost - best_cost, 2),
        "cost_savings_pct": round((default_cost - best_cost) / default_cost * 100, 1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "avg_monthly_income": round(avg_monthly_income, 2),
    }, f, indent=2)

with open(FRONTEND_ASSETS_DIR / "risk_table.json", "w") as f:
    json.dump({"tier_summary": tier_summary, "top_10_highest_risk": top_risk}, f, indent=2)

# Small subset of preprocessing_reference.json that the Predict page's form
# needs (dropdown choices). Kept separate so the frontend never has to read
# from backend/ directly — the two stay independently deployable.
with open(FRONTEND_ASSETS_DIR / "form_options.json", "w") as f:
    json.dump({
        "multi_col_options": multi_col_options,
        "binary_encoders": binary_encoders,
    }, f, indent=2)

print("\nDone. Artifacts written to:")
print(f"  {BACKEND_ARTIFACTS_DIR}")
print(f"  {FRONTEND_ASSETS_DIR}")
