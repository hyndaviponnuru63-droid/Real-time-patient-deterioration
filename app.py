import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.auto_column_mapper import auto_map_columns
from src.data_cleaning import clean_numeric_columns
from src.risk_scoring import calculate_risk, explain_risk, highlight_risk
from src.ml_model import train_model, predict_risk_probability

st.set_page_config(page_title="Patient Deterioration Alert System", layout="wide")
st.title("🩺 Patient Deterioration Alert System")

# Load data
df_raw = pd.read_csv("clinical_data.csv")

if df_raw.empty:
    st.error(" Dataset is empty. Please check the CSV file.")
    st.stop()

st.subheader("📂 Raw Dataset Preview")
st.dataframe(df_raw.head())

# Auto column mapping
df = auto_map_columns(df_raw)

# Data cleaning
df = clean_numeric_columns(df)

# Rule-based risk
df["risk_level"] = df.apply(calculate_risk, axis=1)
df["risk_reason"] = df.apply(explain_risk, axis=1)

# Alerts
high_risk_df = df[df["risk_level"] == "High"]
if not high_risk_df.empty:
    st.error(f"🚨 ALERT: {len(high_risk_df)} High-Risk Patients Detected!")
else:
    st.success("✅ No High-Risk Patients")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", len(df))
col2.metric("High Risk", (df["risk_level"] == "High").sum())
col3.metric("Medium Risk", (df["risk_level"] == "Medium").sum())

# ML prediction
st.subheader("🤖 Machine Learning Risk Prediction")
model, scaler, accuracy, X_train = train_model(df)
df = predict_risk_probability(model, scaler, df)
st.metric("ML Model Accuracy", f"{accuracy*100:.2f}%")

# SHAP explainability
st.subheader("🧠 SHAP Explainability")
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_train)

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_train, show=False)
st.pyplot(fig)

# Download
st.download_button(
    "⬇ Download High-Risk Patients",
    high_risk_df.to_csv(index=False),
    file_name="high_risk_patients.csv",
    mime="text/csv"
)

# Tables
st.subheader("📊 Patient Risk Table")
st.dataframe(df.style.apply(highlight_risk, axis=1))

st.subheader("📈 ML Risk Probability Ranking")
st.dataframe(
    df[["caseid", "risk_level", "risk_probability", "risk_reason"]]
    .sort_values("risk_probability", ascending=False)
)

st.subheader("📊 Risk Distribution")
st.bar_chart(df["risk_level"].value_counts())


