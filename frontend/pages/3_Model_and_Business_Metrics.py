"""
pages/3_Model_and_Business_Metrics.py
========================================
This page carries the most weight for a recruiter evaluating business
judgment, not just modelling skill: WHY the decision threshold was moved
off the default 0.50, and what that was worth in dollars.

Like the EDA page, everything here is precomputed — loaded from
frontend/assets/metrics_figures.json, dataset_summary.json, and
risk_table.json.
"""

import sys
from pathlib import Path

# See pages/1_Predict.py for why this is here.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from utils.styling import inject_base_css, load_asset, render_figure

st.set_page_config(page_title="Model & Business Metrics | Attrition Model", page_icon="💰", layout="wide")
inject_base_css()

summary = load_asset("dataset_summary.json")
metrics = load_asset("metrics_figures.json")
risk = load_asset("risk_table.json")

st.title("💰 Model Performance & Business Impact")

# --- Model card --------------------------------------------------------
st.subheader("Model")
st.markdown(
    "**Logistic Regression** (`class_weight='balanced'`, `C=0.5`) inside a "
    "`StandardScaler` pipeline, validated with **5-fold Stratified Cross-Validation** "
    "to keep the ~16% attrition rate consistent across every fold."
)
cv = summary["cv_metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC (CV mean)", f"{cv['roc_auc']['mean']:.3f}", f"±{cv['roc_auc']['std']:.3f}")
c2.metric("Precision (CV mean)", f"{cv['precision']['mean']:.1%}", f"±{cv['precision']['std']:.1%}")
c3.metric("Recall (CV mean)", f"{cv['recall']['mean']:.1%}", f"±{cv['recall']['std']:.1%}")
c4.metric("F1 (CV mean)", f"{cv['f1']['mean']:.3f}", f"±{cv['f1']['std']:.3f}")

with st.expander("Why Logistic Regression instead of a more complex model?"):
    st.markdown(
        """
        - **Interpretability matters here.** HR and business stakeholders need
          to know *why* someone is flagged, not just that they were. Every
          prediction on the Predict page is explained using the model's real
          coefficients — no black box.
        - **The dataset is small** (1,470 rows). More complex models
          (gradient boosting, neural nets) risk overfitting on this size
          without a much larger validation strategy.
        - Logistic Regression's probability output plugs directly into the
          cost-based threshold optimization below — that's much harder to
          reason about with less-calibrated model types.
        """
    )

st.divider()

# --- The business problem with a default 0.5 threshold --------------------
st.subheader("Why the Default 0.50 Threshold Is the Wrong Choice Here")
st.markdown(
    f"""
    A default classifier flags someone as a flight risk only if the model is
    **more than 50% confident**. That treats a missed leaver (False Negative)
    and a false alarm (False Positive) as equally costly — they aren't:

    - **Missing a real leaver (False Negative)** costs a full replacement:
      recruiting, onboarding, lost productivity — modelled here as
      **6 months of salary**.
    - **Falsely flagging a stayer (False Positive)** costs a wasted retention
      touchpoint — a check-in, maybe a small retention bonus — modelled as
      **0.5 months of salary**, roughly **12× cheaper** than a miss.

    Because a missed leaver is so much more expensive than a false alarm,
    the *optimal* threshold is lower than 0.50 — the model should flag
    people more liberally, on purpose.
    """
)

col1, col2 = st.columns([1.3, 1])
with col1:
    render_figure(metrics["business_cost_curve"], key="cost_curve")
with col2:
    st.metric("Default Threshold (0.50) — Total Cost", f"${summary['default_threshold_cost']:,.0f}")
    st.metric("Optimal Threshold — Total Cost", f"${summary['optimal_threshold_cost']:,.0f}",
              delta=f"-${summary['cost_savings']:,.0f}", delta_color="inverse")
    st.metric("Optimal Threshold", f"{summary['optimal_threshold']:.2f}")
    st.success(f"**{summary['cost_savings_pct']}% cost reduction** by tuning the threshold alone — "
               "no change to the model itself.")

st.divider()

# --- Confusion matrix + curves --------------------------------------------
st.subheader("Model Performance at the Optimal Threshold")
col1, col2, col3 = st.columns(3)
with col1:
    render_figure(metrics["confusion_matrix"], key="cm")
with col2:
    render_figure(metrics["roc_curve"], key="roc")
with col3:
    render_figure(metrics["precision_recall_curve"], key="pr")

cm = summary["confusion_matrix"]
st.markdown(
    f"""
    At threshold **{summary['optimal_threshold']:.2f}**: the model catches
    **{cm['tp']} of {cm['tp'] + cm['fn']} actual leavers** (recall), at the
    cost of **{cm['fp']} false alarms** among stayers. Given the 12:1 cost
    asymmetry above, that trade-off is deliberate — **catching true positives
    is worth far more than avoiding false positives** in this business context.
    """
)

st.divider()

# --- Feature importance -----------------------------------------------------
st.subheader("What Drives the Prediction?")
render_figure(metrics["feature_importance"], key="feat_imp")
st.caption("Green = increases attrition risk, Red = decreases it. Log-odds scale (standardized features).")

st.divider()

# --- Risk tiers and HR action table ----------------------------------------
st.subheader("Employee Risk Tiers — HR Action View")
render_figure(metrics["risk_tier_distribution"], key="tier_dist")

tier_df = pd.DataFrame(risk["tier_summary"])
tier_df = tier_df.rename(columns={
    "Risk_Tier": "Risk Tier", "Employees": "Employees",
    "Avg_Attrition_Prob": "Avg. Probability", "Actual_Leavers": "Actual Leavers",
    "Avg_Replacement_Cost": "Avg. Replacement Cost ($)", "Total_At_Stake": "Total At Stake ($)",
})
st.dataframe(tier_df, use_container_width=True, hide_index=True)

with st.expander("Top 10 highest-risk employees (for HR follow-up)"):
    top_df = pd.DataFrame(risk["top_10_highest_risk"])
    top_df["Attrition_Probability"] = top_df["Attrition_Probability"].map("{:.1%}".format)
    top_df["Replacement_Cost_Est"] = top_df["Replacement_Cost_Est"].map("${:,.0f}".format)
    st.dataframe(top_df, use_container_width=True, hide_index=True)