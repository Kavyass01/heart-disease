import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/updated_version.csv")

df = load_data()

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("📊 Heart Disease Analytics Dashboard")
st.markdown("Comprehensive Healthcare Analytics & Insights")

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
st.subheader("📈 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Patients",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Average Age",
        f"{df['Age'].mean():.1f}"
    )

with col3:
    st.metric(
        "Smokers",
        int(df["Smoking"].sum())
    )

with col4:
    st.metric(
        "Diabetes Cases",
        int(df["Diabetes"].sum())
    )

st.divider()

# --------------------------------------------------
# HEART ATTACK RATE
# --------------------------------------------------
if "Heart Attack" in df.columns:
    attack_rate = round(
        (df["Heart Attack"].sum() / len(df)) * 100,
        2
    )

    st.metric(
        "Heart Attack Rate",
        f"{attack_rate}%"
    )

st.divider()

# --------------------------------------------------
# AGE DISTRIBUTION
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="Age",
        nbins=20,
        title="Age Distribution",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    if "Heart Attack" in df.columns:

        fig = px.pie(
            df,
            names="Heart Attack",
            title="Heart Attack Distribution",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# --------------------------------------------------
# CHOLESTEROL ANALYSIS
# --------------------------------------------------
st.subheader("🩺 Cholesterol Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.box(
        df,
        y="LDL",
        title="LDL Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        df,
        y="HDL",
        title="HDL Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# BLOOD PRESSURE ANALYSIS
# --------------------------------------------------
st.subheader("❤️ Blood Pressure Analysis")

fig = px.scatter(
    df,
    x="Systolic BP",
    y="Diastolic BP",
    color="Heart Attack",
    size="Age",
    hover_data=["Smoking", "Diabetes"],
    title="Blood Pressure Risk Analysis"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# RISK FACTORS
# --------------------------------------------------
st.subheader("🚬 Lifestyle Risk Factors")

col1, col2 = st.columns(2)

with col1:

    smoking_data = (
        df.groupby("Smoking")
        ["Heart Attack"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        smoking_data,
        x="Smoking",
        y="Heart Attack",
        title="Smoking vs Heart Attack Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    diabetes_data = (
        df.groupby("Diabetes")
        ["Heart Attack"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        diabetes_data,
        x="Diabetes",
        y="Heart Attack",
        title="Diabetes vs Heart Attack Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------
st.subheader("🔥 Correlation Matrix")

corr = df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Feature Correlation Heatmap"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
st.subheader("📄 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Dataset",
    data=csv,
    file_name="heart_disease_data.csv",
    mime="text/csv"
)
