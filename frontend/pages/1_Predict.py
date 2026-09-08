"""
pages/1_Predict.py
===================
Takes one employee's details from a form, sends them to the FastAPI
/predict endpoint, and displays the probability, risk tier, and the
factors that pushed the prediction up or down.

This page holds NO model logic itself — it only collects input and
displays what the API returns. That separation is the whole point of
having a backend.
"""

import sys
from pathlib import Path

# Streamlit Cloud sometimes runs pages/*.py without the frontend/ directory
# on sys.path (works locally because `streamlit run app.py` adds it
# automatically). This makes `from utils....` resolve reliably either way.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from utils.api_client import get_prediction
from utils.styling import TIER_COLORS, inject_base_css, load_asset, risk_badge

st.set_page_config(page_title="Predict | Attrition Model", page_icon="🎯", layout="wide")
inject_base_css()

form_options = load_asset("form_options.json")
DEPARTMENT_OPTS = form_options["multi_col_options"]["Department"]
JOBROLE_OPTS = form_options["multi_col_options"]["JobRole"]
EDUFIELD_OPTS = form_options["multi_col_options"]["EducationField"]
MARITAL_OPTS = form_options["multi_col_options"]["MaritalStatus"]
TRAVEL_OPTS = form_options["multi_col_options"]["BusinessTravel"]
GENDER_OPTS = form_options["binary_encoders"]["Gender"]
OVERTIME_OPTS = form_options["binary_encoders"]["OverTime"]

st.title("🎯 Predict Attrition Risk")
st.caption("Fill in an employee's details and get a live prediction from the deployed model.")

with st.form("prediction_form"):
    st.subheader("Demographics")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 18, 60, 30)
    gender = c2.selectbox("Gender", GENDER_OPTS)
    marital_status = c3.selectbox("Marital Status", MARITAL_OPTS)
    c4, c5 = st.columns(2)
    distance = c4.slider("Distance From Home (miles)", 0, 30, 8)
    education = c5.slider("Education Level (1=Below College, 5=Doctor)", 1, 5, 3)
    education_field = st.selectbox("Education Field", EDUFIELD_OPTS)

    st.subheader("Job Info")
    c1, c2, c3 = st.columns(3)
    department = c1.selectbox("Department", DEPARTMENT_OPTS)
    job_role = c2.selectbox("Job Role", JOBROLE_OPTS)
    job_level = c3.slider("Job Level", 1, 5, 2)
    c4, c5 = st.columns(2)
    business_travel = c4.selectbox("Business Travel", TRAVEL_OPTS, index=1)
    overtime = c5.selectbox("Works OverTime?", OVERTIME_OPTS)
    c6, c7, c8, c9 = st.columns(4)
    num_companies = c6.number_input("Companies Worked At", 0, 9, 2)
    total_years = c7.number_input("Total Working Years", 0, 40, 8)
    years_at_company = c8.number_input("Years At This Company", 0, 40, 4)
    years_in_role = c9.number_input("Years In Current Role", 0, 18, 2)
    c10, c11, c12 = st.columns(3)
    years_since_promo = c10.number_input("Years Since Last Promotion", 0, 15, 1)
    years_with_manager = c11.number_input("Years With Current Manager", 0, 17, 2)
    trainings = c12.number_input("Trainings Last Year", 0, 6, 2)

    st.subheader("Compensation")
    c1, c2, c3, c4 = st.columns(4)
    monthly_income = c1.number_input("Monthly Income ($)", 1000, 20000, 5000, step=100)
    daily_rate = c2.number_input("Daily Rate", 100, 1500, 800)
    hourly_rate = c3.number_input("Hourly Rate", 30, 100, 65)
    monthly_rate = c4.number_input("Monthly Rate", 2000, 27000, 15000, step=100)
    c5, c6, c7 = st.columns(3)
    salary_hike = c5.slider("% Salary Hike Last Review", 11, 25, 14)
    stock_option = c6.slider("Stock Option Level", 0, 3, 0)
    performance = c7.slider("Performance Rating", 3, 4, 3)

    st.subheader("Satisfaction & Work-Life Scores (1=Low, 4=Very High)")
    c1, c2, c3, c4, c5 = st.columns(5)
    env_sat = c1.slider("Environment Satisfaction", 1, 4, 3)
    job_sat = c2.slider("Job Satisfaction", 1, 4, 3)
    rel_sat = c3.slider("Relationship Satisfaction", 1, 4, 3)
    wlb = c4.slider("Work-Life Balance", 1, 4, 3)
    job_involvement = c5.slider("Job Involvement", 1, 4, 3)

    submitted = st.form_submit_button("Predict Attrition Risk", use_container_width=True, type="primary")

if submitted:
    employee_payload = {
        "Age": age, "Gender": gender, "MaritalStatus": marital_status,
        "DistanceFromHome": distance, "Education": education, "EducationField": education_field,
        "Department": department, "JobRole": job_role, "JobLevel": job_level,
        "BusinessTravel": business_travel, "OverTime": overtime,
        "NumCompaniesWorked": num_companies, "TotalWorkingYears": total_years,
        "YearsAtCompany": years_at_company, "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_since_promo, "YearsWithCurrManager": years_with_manager,
        "TrainingTimesLastYear": trainings,
        "MonthlyIncome": monthly_income, "DailyRate": daily_rate, "HourlyRate": hourly_rate,
        "MonthlyRate": monthly_rate, "PercentSalaryHike": salary_hike,
        "StockOptionLevel": stock_option, "PerformanceRating": performance,
        "EnvironmentSatisfaction": env_sat, "JobSatisfaction": job_sat,
        "RelationshipSatisfaction": rel_sat, "WorkLifeBalance": wlb,
        "JobInvolvement": job_involvement,
    }

    try:
        with st.spinner("Calling the prediction API..."):
            result = get_prediction(employee_payload)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.divider()
    st.subheader("Result")

    tier = result["risk_tier"]
    prob = result["attrition_probability"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Attrition Probability", f"{prob:.1%}")
    col2.markdown(f"### {risk_badge(tier)}")
    col3.metric("Decision Threshold Used", f"{result['decision_threshold_used']:.2f}")

    st.progress(min(prob, 1.0), text=f"{prob:.1%} predicted probability of leaving")

    action_copy = {
        "High Risk": "Recommend a retention conversation with this employee's manager this week.",
        "Medium Risk": "Worth monitoring — check in during the next 1:1.",
        "Low Risk": "No action needed based on current signals.",
    }
    st.info(action_copy.get(tier, ""))

    st.divider()
    col_up, col_down = st.columns(2)
    with col_up:
        st.markdown("**⬆️ Factors increasing risk**")
        for f in result["top_risk_factors"]:
            st.markdown(f"- `{f['feature']}` (+{f['contribution']:.2f})")
    with col_down:
        st.markdown("**⬇️ Factors decreasing risk**")
        for f in result["top_protective_factors"]:
            st.markdown(f"- `{f['feature']}` ({f['contribution']:.2f})")

    st.caption(
        "Factor values are each feature's exact contribution to the model's "
        "log-odds output (scaled input × logistic regression coefficient) — "
        "not an approximation."
    )