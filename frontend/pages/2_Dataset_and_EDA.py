"""
pages/2_Dataset_and_EDA.py
============================
Everything here comes from frontend/assets/eda_figures.json and
dataset_summary.json — both precomputed once by scripts/export_artifacts.py.
This page never touches the raw CSV or recomputes a single statistic;
it only loads JSON and renders it.
"""

import streamlit as st

from utils.styling import inject_base_css, load_asset, render_figure

st.set_page_config(page_title="Dataset & EDA | Attrition Model", page_icon="📈", layout="wide")
inject_base_css()

summary = load_asset("dataset_summary.json")
eda = load_asset("eda_figures.json")

st.title("📈 Dataset & Exploratory Analysis")
st.caption("IBM HR Analytics Employee Attrition dataset (Kaggle)")

# --- Dataset overview ------------------------------------------------------
st.subheader("Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw Rows × Columns", f"{summary['raw_shape'][0]} × {summary['raw_shape'][1]}")
c2.metric("After Cleaning", f"{summary['shape_after_cleaning'][0]} × {summary['shape_after_cleaning'][1]}")
c3.metric("Missing Values", summary["missing_values_total"])
c4.metric("Attrition Rate", f"{summary['attrition_rate_pct']}%")

with st.expander("Why 4 columns were dropped before modelling"):
    st.markdown(
        "These columns carried **zero predictive signal** — either every "
        "employee had the same value, or the column was just a row ID:"
    )
    for col, info in summary["constant_cols_dropped"].items():
        if info["type"] == "constant":
            st.markdown(f"- **{col}** — every employee has the same value (`{info['value']}`)")
        else:
            st.markdown(f"- **{col}** — a unique row identifier ({info['n_unique']:,} distinct values, no signal)")
    st.markdown(
        f"No missing values were present in any of the remaining "
        f"{summary['shape_after_cleaning'][1]} columns, so no imputation was needed."
    )

st.divider()

# --- Target distribution ---------------------------------------------------
st.subheader("Target Distribution — Why This Is an Imbalanced Classification Problem")
col1, col2 = st.columns([1, 1.4])
with col1:
    render_figure(eda["target_distribution"], key="target_dist")
with col2:
    st.markdown(
        f"""
        Only **{summary['attrition_rate_pct']}%** of employees left — roughly a
        **5:1 imbalance**. This matters a lot for modelling choices made later:

        - Accuracy alone would be a misleading metric — a model that always
          predicts "stays" would already be ~84% "accurate" while catching
          zero leavers.
        - The model uses `class_weight='balanced'` so the minority class
          (leavers) isn't ignored during training.
        - Evaluation leans on **ROC-AUC, Precision, and Recall** instead of
          accuracy — see the Business Metrics page.
        """
    )

st.divider()

# --- Act 1: Where is attrition concentrated? -------------------------------
st.subheader("Act 1 — Where Is Attrition Concentrated?")
col1, col2, col3 = st.columns(3)
with col1:
    render_figure(eda["attrition_by_department"], key="dept")
with col2:
    render_figure(eda["attrition_by_role"], key="role")
with col3:
    render_figure(eda["overtime_vs_attrition"], key="overtime")
st.caption(
    "OverTime turns out to be the single strongest raw signal in the dataset — "
    "employees who work overtime leave at a dramatically higher rate."
)

st.divider()

# --- Act 2: Signal extraction -----------------------------------------------
st.subheader("Act 2 — What Features Drive Flight Risk?")
col1, col2, col3 = st.columns(3)
with col1:
    render_figure(eda["income_distribution"], key="income")
with col2:
    render_figure(eda["promotion_gap"], key="promo")
with col3:
    render_figure(eda["top_correlations"], key="corr")
st.caption(
    "Leavers skew toward lower income and longer gaps since their last "
    "promotion — both feed directly into the engineered features used by the model."
)
