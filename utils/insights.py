import pandas as pd


def calculate_heart_attack_rate(df):
    """
    Calculate heart attack percentage
    """
    rate = (df["Heart Attack"].sum() / len(df)) * 100
    return round(rate, 2)


def get_population_statistics(df):
    """
    Generate basic dataset statistics
    """

    stats = {
        "Total Patients": len(df),
        "Average Age": round(df["Age"].mean(), 1),
        "Average Cholesterol": round(
            df["Total Cholesterol"].mean(), 1
        ),
        "Average LDL": round(
            df["LDL"].mean(), 1
        ),
        "Average HDL": round(
            df["HDL"].mean(), 1
        ),
        "Heart Attack Rate": calculate_heart_attack_rate(df)
    }

    return stats


def feature_importance_from_correlation(df):
    """
    Estimate feature importance using correlation
    """

    target = "Heart Attack"

    features = [
        col
        for col in df.columns
        if col != target
    ]

    importance = []

    for col in features:

        if pd.api.types.is_numeric_dtype(df[col]):

            corr = abs(
                df[col].corr(df[target])
            )

            importance.append(
                {
                    "Feature": col,
                    "Importance": round(corr, 4)
                }
            )

    importance_df = pd.DataFrame(
        importance
    ).sort_values(
        by="Importance",
        ascending=False
    )

    return importance_df


def get_top_risk_factors(df, top_n=5):
    """
    Return top correlated risk factors
    """

    importance_df = feature_importance_from_correlation(df)

    return importance_df.head(top_n)


def generate_health_insights(df):
    """
    Generate AI-style health insights
    """

    insights = []

    if df["Age"].mean() > 50:
        insights.append(
            "Average patient age exceeds 50 years, indicating elevated cardiovascular risk."
        )

    if df["Smoking"].mean() > 0.30:
        insights.append(
            "Smoking prevalence is high and may contribute significantly to heart attack occurrence."
        )

    if df["Diabetes"].mean() > 0.20:
        insights.append(
            "Diabetes is common within the population and should be monitored closely."
        )

    if df["LDL"].mean() > 130:
        insights.append(
            "Average LDL cholesterol is above recommended levels."
        )

    if df["HDL"].mean() < 40:
        insights.append(
            "Average HDL cholesterol is below the healthy threshold."
        )

    if df["Systolic BP"].mean() > 130:
        insights.append(
            "Elevated systolic blood pressure suggests widespread hypertension risk."
        )

    if len(insights) == 0:
        insights.append(
            "No major population-wide cardiovascular concerns detected."
        )

    return insights


def generate_recommendations(df):
    """
    Generate health recommendations
    """

    recommendations = []

    if df["Smoking"].mean() > 0.20:
        recommendations.append(
            "Implement smoking cessation programs."
        )

    if df["LDL"].mean() > 130:
        recommendations.append(
            "Encourage dietary interventions to reduce LDL cholesterol."
        )

    if df["Systolic BP"].mean() > 130:
        recommendations.append(
            "Monitor blood pressure regularly and promote exercise."
        )

    if df["Diabetes"].mean() > 0.15:
        recommendations.append(
            "Strengthen diabetes screening and management."
        )

    recommendations.extend([
        "Promote regular cardiovascular screening.",
        "Encourage healthy lifestyle habits.",
        "Increase awareness about heart disease prevention."
    ])

    return recommendations


def classify_patient_risk(patient):
    """
    Rule-based risk classification
    """

    score = 0

    if patient["Age"] > 60:
        score += 20

    if patient["Total Cholesterol"] > 240:
        score += 20

    if patient["LDL"] > 160:
        score += 20

    if patient["Systolic BP"] > 140:
        score += 15

    if patient["Smoking"] == 1:
        score += 15

    if patient["Diabetes"] == 1:
        score += 10

    if score < 30:
        category = "Low Risk"

    elif score < 60:
        category = "Moderate Risk"

    else:
        category = "High Risk"

    return {
        "Risk Score": score,
        "Risk Category": category
    }


def dataset_summary(df):
    """
    Generate complete summary dictionary
    """

    return {
        "statistics": get_population_statistics(df),
        "heart_attack_rate": calculate_heart_attack_rate(df),
        "top_risk_factors": get_top_risk_factors(df),
        "insights": generate_health_insights(df),
        "recommendations": generate_recommendations(df)
    }
