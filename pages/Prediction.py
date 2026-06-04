import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction")

@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None

model, scaler = load_model()

if model is None:
    st.stop()

st.title("Heart Disease Prediction")

age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex", [0, 1])
cp = st.number_input("Chest Pain Type", 0, 3, 0)
trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
chol = st.number_input("Cholesterol", 50, 700, 200)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.number_input("Rest ECG", 0, 2, 0)
thalach = st.number_input("Max Heart Rate", 50, 250, 150)
exang = st.selectbox("Exercise Angina", [0, 1])
oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0)
slope = st.number_input("Slope", 0, 2, 1)
ca = st.number_input("CA", 0, 4, 0)
thal = st.number_input("Thal", 0, 3, 1)

if st.button("Predict"):
    data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")
