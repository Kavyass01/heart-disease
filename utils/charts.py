import plotly.express as px
import plotly.graph_objects as go


def age_distribution_chart(df):
    """
    Age Distribution Histogram
    """
    fig = px.histogram(
        df,
        x="Age",
        nbins=20,
        title="Age Distribution",
        template="plotly_white"
    )

    return fig


def heart_attack_pie_chart(df):
    """
    Heart Attack Distribution
    """
    fig = px.pie(
        df,
        names="Heart Attack",
        title="Heart Attack Distribution",
        hole=0.4
    )

    return fig


def ldl_boxplot(df):
    """
    LDL Distribution
    """
    fig = px.box(
        df,
        y="LDL",
        title="LDL Distribution"
    )

    return fig


def hdl_boxplot(df):
    """
    HDL Distribution
    """
    fig = px.box(
        df,
        y="HDL",
        title="HDL Distribution"
    )

    return fig


def cholesterol_vs_attack(df):
    """
    Cholesterol vs Heart Attack
    """
    fig = px.box(
        df,
        x="Heart Attack",
        y="Total Cholesterol",
        color="Heart Attack",
        title="Cholesterol vs Heart Attack"
    )

    return fig


def blood_pressure_scatter(df):
    """
    Blood Pressure Analysis
    """
    fig = px.scatter(
        df,
        x="Systolic BP",
        y="Diastolic BP",
        color="Heart Attack",
        size="Age",
        title="Blood Pressure Analysis",
        hover_data=["Smoking", "Diabetes"]
    )

    return fig


def smoking_risk_chart(df):
    """
    Smoking vs Heart Attack Risk
    """
    risk = (
        df.groupby("Smoking")["Heart Attack"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        risk,
        x="Smoking",
        y="Heart Attack",
        title="Smoking vs Heart Attack Risk"
    )

    return fig


def diabetes_risk_chart(df):
    """
    Diabetes vs Heart Attack Risk
    """
    risk = (
        df.groupby("Diabetes")["Heart Attack"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        risk,
        x="Diabetes",
        y="Heart Attack",
        title="Diabetes vs Heart Attack Risk"
    )

    return fig


def correlation_heatmap(df):
    """
    Correlation Matrix
    """
    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    return fig


def feature_importance_chart(importance_df):
    """
    Feature Importance Chart
    """
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    return fig


def risk_gauge(score):
    """
    Prediction Risk Gauge
    """

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Heart Attack Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30]},
                {"range": [30, 70]},
                {"range": [70, 100]}
            ]
        }
    ))

    fig.update_layout(height=350)

    return fig


def age_vs_attack_chart(df):
    """
    Heart Attack by Age
    """
    fig = px.histogram(
        df,
        x="Age",
        color="Heart Attack",
        nbins=25,
        title="Heart Attack Distribution by Age"
    )

    return fig


def cholesterol_histogram(df):
    """
    Cholesterol Distribution
    """
    fig = px.histogram(
        df,
        x="Total Cholesterol",
        nbins=30,
        title="Cholesterol Distribution"
    )

    return fig


def ldl_hdl_scatter(df):
    """
    HDL vs LDL Analysis
    """
    fig = px.scatter(
        df,
        x="HDL",
        y="LDL",
        color="Heart Attack",
        title="HDL vs LDL Analysis"
    )

    return fig
