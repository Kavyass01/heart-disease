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
    df = pd.read_csv("data/updated_version.csv")
    df.columns = df.columns.str.strip()
    return df

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
        f"{df['age'].mean():.1f}"
    )

with col3:
    st.metric(
        "Smokers",
        int(df["smoking"].sum())
    )

with col4:
    st.metric(
        "Diabetes Cases",
        int(df["diabetes"].sum())
    )

st.divider()

# --------------------------------------------------
# HEART ATTACK RATE
# --------------------------------------------------
if "heart_attack" in df.columns:

    attack_rate = round(
        (df["heart_attack"].sum() / len(df)) * 100,
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
        x="age",
        nbins=20,
        title="Age Distribution",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    if "heart_attack" in df.columns:

        fig = px.pie(
            df,
            names="heart_attack",
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
        y="ldl",
        title="LDL Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        df,
        y="hdl",
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
    x="systolic_bp",
    y="diastolic_bp",
    color="heart_attack",
    size="age",
    hover_data=["smoking", "diabetes"],
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
        df.groupby("smoking")["heart_attack"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        smoking_data,
        x="smoking",
        y="heart_attack",
        title="Smoking vs Heart Attack Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    diabetes_data = (
        df.groupby("diabetes")["heart_attack"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        diabetes_data,
        x="diabetes",
        y="heart_attack",
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
