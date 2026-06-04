import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI Insights",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/updated_version.csv")

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Show actual column names for debugging
    st.write("Detected Columns:")
    st.write(df.columns.tolist())

    return df
# -----------------------------------
# OVERVIEW METRICS
# -----------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Patients",
        len(df)
    )

with col2:
    st.metric(
        "Avg Age",
        round(df["Age"].mean(), 1)
    )

with col3:
    st.metric(
        "Avg Cholesterol",
        round(df["Total Cholesterol"].mean(), 1)
    )

with col4:
    st.metric(
        "Heart Attack Cases",
        int(df["Heart Attack"].sum())
    )

st.divider()

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------
st.subheader("📊 AI Feature Importance")

features = [
    "Age",
    "Total Cholesterol",
    "LDL",
    "HDL",
    "Systolic BP",
    "Diastolic BP",
    "Smoking",
    "Diabetes"
]

importance = []

for col in features:
    corr = abs(df[col].corr(df["Heart Attack"]))
    importance.append(corr)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(
    by="Importance",
    ascending=False
)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Estimated Feature Importance"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------
# RISK SEGMENTATION
# -----------------------------------
st.subheader("🎯 Risk Segmentation")

df["Risk Category"] = pd.cut(
    df["Age"],
    bins=[0, 40, 60, 100],
    labels=["Low", "Medium", "High"]
)

risk_data = (
    df.groupby("Risk Category")
    ["Heart Attack"]
    .mean()
    .reset_index()
)

fig = px.pie(
    risk_data,
    names="Risk Category",
    values="Heart Attack",
    title="Risk Category Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------
# TOP RISK FACTORS
# -----------------------------------
st.subheader("🚨 Top Risk Factors")

top_factors = importance_df.head(5)

for i, row in top_factors.iterrows():

    st.info(
        f"**{row['Feature']}** has a strong relationship with Heart Attack occurrence."
    )

# -----------------------------------
# HEALTH INSIGHTS
# -----------------------------------
st.subheader("🧠 Automated Health Insights")

insights = []

if df["Age"].mean() > 50:
    insights.append(
        "Average patient age is above 50, indicating elevated cardiovascular risk."
    )

if df["Smoking"].mean() > 0.3:
    insights.append(
        "Smoking prevalence is relatively high within the dataset."
    )

if df["Diabetes"].mean() > 0.2:
    insights.append(
        "Diabetes appears as a significant health concern."
    )

if df["LDL"].mean() > 130:
    insights.append(
        "Average LDL cholesterol exceeds recommended levels."
    )

if df["Systolic BP"].mean() > 130:
    insights.append(
        "Average systolic blood pressure is elevated."
    )

if len(insights) == 0:
    st.success(
        "No major population-wide health concerns detected."
    )

for insight in insights:
    st.write("✅", insight)

# -----------------------------------
# HEART ATTACK ANALYSIS
# -----------------------------------
st.subheader("❤️ Heart Attack Analysis")

attack_rate = (
    df["Heart Attack"].sum() / len(df)
) * 100

st.metric(
    "Heart Attack Rate",
    f"{attack_rate:.2f}%"
)

fig = px.histogram(
    df,
    x="Age",
    color="Heart Attack",
    nbins=25,
    title="Heart Attack Distribution by Age"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------
# RECOMMENDATIONS
# -----------------------------------
st.subheader("📋 AI Recommendations")

recommendations = [
    "Promote smoking cessation programs.",
    "Monitor LDL cholesterol levels regularly.",
    "Encourage physical activity and healthy diets.",
    "Implement routine cardiovascular screening.",
    "Focus on diabetic patient monitoring.",
    "Monitor blood pressure and reduce hypertension risk."
]

for rec in recommendations:
    st.success(rec)

# -----------------------------------
# DATA PREVIEW
# -----------------------------------
st.subheader("📄 Dataset Snapshot")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.caption(
    "AI Insights Module • Heart Disease Analytics Platform"
)
