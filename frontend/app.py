"""
app.py
======
Entry point for the Streamlit app. This file becomes the "Overview" page;
the other three pages live in frontend/pages/ and Streamlit's multipage
navigation picks them up automatically (shown in the sidebar).

Run with:  streamlit run app.py
"""

import streamlit as st

from utils.api_client import check_backend_health
from utils.styling import inject_base_css, load_asset

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide",
)
inject_base_css()

summary = load_asset("dataset_summary.json")

# --- Header -----------------------------------------------------------
st.title("📊 Employee Attrition Early-Warning System")
st.markdown(
    "A machine learning system that predicts which employees are at risk "
    "of leaving — and translates that prediction into a business cost, "
    "not just an accuracy score."
)

health = check_backend_health()
if health and "optimal_threshold" in health:
    st.success(f"Prediction API is live — optimal decision threshold: {health['optimal_threshold']:.2f}")
elif health:
    st.warning(
        "Prediction API responded, but with an unexpected format. "
        "Double-check BACKEND_URL points at the right deployed backend."
    )
else:
    st.warning(
        "Prediction API is unreachable right now. The Overview, Dataset & EDA, "
        "and Business Metrics pages will still work (they use precomputed data) — "
        "only the live Predict page needs the API."
    )

st.divider()

# --- Headline metrics ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees Analyzed", f"{summary['raw_shape'][0]:,}")
col2.metric("Attrition Rate", f"{summary['attrition_rate_pct']}%")
col3.metric("Model ROC-AUC (5-fold CV)", f"{summary['cv_metrics']['roc_auc']['mean']:.3f}")
col4.metric("Cost Saved by Threshold Tuning", f"${summary['cost_savings']:,.0f}")

st.divider()

# --- Problem statement ---------------------------------------------------
st.subheader("The Problem")
st.markdown(
    """
    Losing an employee typically costs **6 months of their salary** in
    recruiting, onboarding, and lost productivity. Most attrition models
    stop at "predict yes or no" — this project goes further and asks:
    **at what point does flagging someone as a flight risk actually save
    the company money, versus just creating false alarms?**
    """
)

st.subheader("Approach")
approach_cols = st.columns(4)
steps = [
    ("1. Explore", "3-act visual analysis of where attrition concentrates — "
                    "department, role, overtime, income, tenure."),
    ("2. Engineer", "Encode categoricals, build Satisfaction_Score, "
                     "Tenure_Ratio, and Income_vs_Role_Median features."),
    ("3. Model", "Logistic Regression with class-balanced weights, "
                  "validated with 5-fold Stratified CV."),
    ("4. Optimize", "Sweep the decision threshold against a real cost "
                     "function — not just accuracy — to find the "
                     "business-optimal cutoff."),
]
for col, (title, desc) in zip(approach_cols, steps):
    with col:
        st.markdown(f"**{title}**")
        st.caption(desc)

st.divider()

# --- Tech stack -----------------------------------------------------------
st.subheader("Tech Stack")
st.markdown(
    "`Python` `scikit-learn` `pandas` `FastAPI` `Pydantic` `Streamlit` `Plotly`"
)

st.divider()
st.subheader("Explore the Project")
nav_cols = st.columns(3)
with nav_cols[0]:
    st.page_link("pages/1_Predict.py", label="🎯 Try a Live Prediction", use_container_width=True)
with nav_cols[1]:
    st.page_link("pages/2_Dataset_and_EDA.py", label="📈 Dataset & EDA", use_container_width=True)
with nav_cols[2]:
    st.page_link("pages/3_Model_and_Business_Metrics.py", label="💰 Model & Business Metrics", use_container_width=True)