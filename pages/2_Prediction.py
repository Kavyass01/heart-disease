import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------------
# LOAD DATA & TRAIN MODEL
# -----------------------------------
@st.cache_resource
def load_model():

    df = pd.read_csv("data/updated_version.csv")
    df.columns = df.columns.str.strip()

    X = df.drop(columns=["heart_attack"])
    y = df["heart_attack"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler


model, scaler = load_model()

# -----------------------------------
# TITLE
# -----------------------------------
st.title("❤️ Heart Disease Prediction")

st.markdown(
    "Enter patient information below to estimate heart attack risk."
)

st.divider()

# -----------------------------------
# INPUTS
# -----------------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=30,
        max_value=100,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        [0, 1],
        help="0 = Female, 1 = Male"
    )

    total_cholesterol = st.number_input(
        "Total Cholesterol",
        min_value=100,
        max_value=400,
        value=200
    )

    ldl = st.number_input(
        "LDL",
        min_value=50,
        max_value=300,
        value=120
    )

    hdl = st.number_input(
        "HDL",
        min_value=20,
        max_value=100,
        value=50
    )

with col2:

    systolic_bp = st.number_input(
        "Systolic BP",
        min_value=80,
        max_value=250,
        value=120
    )

    diastolic_bp = st.number_input(
        "Diastolic BP",
        min_value=50,
        max_value=150,
        value=80
    )

    smoking = st.selectbox(
        "Smoking",
        [0, 1],
        help="0 = No, 1 = Yes"
    )

    diabetes = st.selectbox(
        "Diabetes",
        [0, 1],
        help="0 = No, 1 = Yes"
    )

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("Predict Risk"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "total_cholesterol": [total_cholesterol],
        "ldl": [ldl],
        "hdl": [hdl],
        "systolic_bp": [systolic_bp],
        "diastolic_bp": [diastolic_bp],
        "smoking": [smoking],
        "diabetes": [diabetes]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ High Risk of Heart Attack ({probability*100:.1f}%)"
        )
    else:
        st.success(
            f"✅ Low Risk of Heart Attack ({(1-probability)*100:.1f}%)"
        )

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.caption(
    "Heart Disease Analytics Platform"
)
