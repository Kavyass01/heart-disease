import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Heart Attack Prediction",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------------
# LOAD MODEL & SCALER
# ----------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ----------------------------------
# TITLE
# ----------------------------------
st.title("🩺 Heart Attack Risk Prediction")
st.markdown("Enter patient health information to estimate heart attack risk.")

# ----------------------------------
# INPUT FORM
# ----------------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=45
    )

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    total_cholesterol = st.number_input(
        "Total Cholesterol",
        min_value=100,
        max_value=500,
        value=200
    )

    ldl = st.number_input(
        "LDL",
        min_value=30,
        max_value=300,
        value=120
    )

    hdl = st.number_input(
        "HDL",
        min_value=10,
        max_value=120,
        value=50
    )

with col2:

    systolic_bp = st.number_input(
        "Systolic BP",
        min_value=70,
        max_value=250,
        value=120
    )

    diastolic_bp = st.number_input(
        "Diastolic BP",
        min_value=40,
        max_value=180,
        value=80
    )

    smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )

    diabetes = st.selectbox(
        "Diabetes",
        ["No", "Yes"]
    )

# ----------------------------------
# PREPROCESS INPUT
# ----------------------------------
sex = 1 if sex == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0

# ----------------------------------
# PREDICT BUTTON
# ----------------------------------
if st.button("🔍 Predict Risk", use_container_width=True):

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "Total Cholesterol": [total_cholesterol],
        "LDL": [ldl],
        "HDL": [hdl],
        "Systolic BP": [systolic_bp],
        "Diastolic BP": [diastolic_bp],
        "Smoking": [smoking],
        "Diabetes": [diabetes]
    })

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    risk_percentage = round(probability * 100, 2)

    st.divider()

    # ----------------------------------
    # RESULTS
    # ----------------------------------
    col1, col2 = st.columns([1, 1])

    with col1:

        st.metric(
            "Risk Probability",
            f"{risk_percentage}%"
        )

        if risk_percentage < 30:
            st.success("🟢 Low Risk")
        elif risk_percentage < 70:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")

    with col2:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percentage,
            title={"text": "Heart Attack Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 30]},
                    {"range": [30, 70]},
                    {"range": [70, 100]}
                ]
            }
        ))

        fig.update_layout(height=350)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ----------------------------------
    # HEALTH INSIGHTS
    # ----------------------------------
    st.subheader("📋 Health Insights")

    insights = []

    if age > 60:
        insights.append("Age is a significant cardiovascular risk factor.")

    if total_cholesterol > 240:
        insights.append("High cholesterol level detected.")

    if ldl > 160:
        insights.append("LDL cholesterol is above recommended levels.")

    if hdl < 40:
        insights.append("Low HDL cholesterol may increase heart risk.")

    if systolic_bp > 140:
        insights.append("Elevated systolic blood pressure detected.")

    if smoking == 1:
        insights.append("Smoking significantly increases heart disease risk.")

    if diabetes == 1:
        insights.append("Diabetes is a major cardiovascular risk factor.")

    if len(insights) == 0:
        st.success("No major risk factors identified.")

    for item in insights:
        st.write("•", item)

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")
st.caption("Heart Disease Analytics & Prediction System")
