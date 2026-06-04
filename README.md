# ❤️ Heart Disease Analytics & Risk Prediction Platform

An end-to-end Machine Learning and Data Analytics application built with Python and Streamlit for predicting heart attack risk and providing interactive healthcare insights.

---

## 🚀 Project Overview

This project combines:

* 📊 Interactive Healthcare Analytics Dashboard
* 🤖 Machine Learning-Based Heart Attack Prediction
* 📈 Advanced Data Visualization
* 🔍 Explainable AI (SHAP)
* 🎯 Risk Segmentation & Insights
* ☁️ Streamlit Cloud Deployment

The platform helps healthcare professionals and researchers analyze patient health indicators and predict the likelihood of heart attacks using machine learning models.

---

# Dataset Features

| Feature           | Description                   |
| ----------------- | ----------------------------- |
| Age               | Patient Age                   |
| Sex               | Gender (0 = Female, 1 = Male) |
| Total Cholesterol | Total Cholesterol Level       |
| LDL               | Low Density Lipoprotein       |
| HDL               | High Density Lipoprotein      |
| Systolic BP       | Systolic Blood Pressure       |
| Diastolic BP      | Diastolic Blood Pressure      |
| Smoking           | Smoking Status                |
| Diabetes          | Diabetes Status               |
| Heart Attack      | Target Variable               |

---

# Project Structure

```text
heart-disease-analytics/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── updated_version.csv
│
├── models/
│   ├── train_model.py
│   ├── model.pkl
│   └── scaler.pkl
│
├── pages/
│   ├── 1_Analytics_Dashboard.py
│   ├── 2_Prediction.py
│   └── 3_AI_Insights.py
│
├── utils/
│   ├── preprocessing.py
│   ├── charts.py
│   └── insights.py
│
└── assets/
    ├── logo.png
    └── banner.jpg
```

---

# Features

## 📊 Analytics Dashboard

* Dataset Overview
* KPI Cards
* Age Distribution Analysis
* Cholesterol Analysis
* Blood Pressure Analysis
* Smoking vs Heart Attack Analysis
* Diabetes vs Heart Attack Analysis
* Correlation Heatmap

---

## 🤖 Prediction Module

Users can input:

* Age
* Gender
* Cholesterol Levels
* Blood Pressure
* Smoking Status
* Diabetes Status

The system predicts:

* Heart Attack Risk
* Probability Score
* Risk Category

---

## 📈 Machine Learning Models

Implemented Models:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

---

## 🔍 Explainable AI

The application uses SHAP to explain:

* Feature Importance
* Individual Predictions
* Global Model Behavior

---

# Technology Stack

| Technology   | Usage                     |
| ------------ | ------------------------- |
| Python       | Development               |
| Pandas       | Data Processing           |
| NumPy        | Numerical Computing       |
| Scikit-Learn | Machine Learning          |
| XGBoost      | Advanced Prediction       |
| Plotly       | Interactive Visualization |
| Streamlit    | Web Application           |
| SHAP         | Explainable AI            |
| Joblib       | Model Serialization       |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/heart-disease-analytics.git

cd heart-disease-analytics
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
streamlit run app.py
```

Application will start at:

```text
http://localhost:8501
```

---

# Model Training

To retrain the model:

```bash
cd models

python train_model.py
```

Generated files:

```text
model.pkl
scaler.pkl
```

---

# Deployment

## Streamlit Cloud

1. Push code to GitHub
2. Open Streamlit Community Cloud
3. Connect GitHub Repository
4. Select:

   * Repository
   * Branch = main
   * Main File = app.py
5. Deploy

---

# Future Enhancements

* PDF Health Reports
* Doctor Recommendation Engine
* Real-Time Health Monitoring
* Deep Learning Models
* Mobile Responsive UI
* Multi-Disease Prediction
* Healthcare Chatbot Integration
* Docker Deployment
* CI/CD Pipeline

---

# Sample Dashboard

### Analytics

* KPI Metrics
* Trend Analysis
* Correlation Analysis
* Risk Segmentation

### Prediction

* Real-Time Heart Attack Prediction
* Probability Score
* Explainable Results

---

# Results

The Random Forest model achieved high predictive performance and provides interpretable insights through SHAP visualizations.

Key risk indicators identified:

* Age
* LDL Cholesterol
* Systolic Blood Pressure
* Smoking
* Diabetes

---

# Author

Kavya S S

Healthcare Analytics | Data Science | Machine Learning | Streamlit Development

---

## ⭐ If you found this project useful, please give it a star on GitHub!
