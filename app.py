import streamlit as st
import pandas as pd

from src.risk_scoring import (
    calculate_risk,
    explain_risk,
    calculate_news,
    highlight_risk
)

st.set_page_config(page_title="Patient Deterioration Alert System", layout="wide")

st.title("🩺 Patient Deterioration Alert System")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("clinical_data.csv")

# Convert required columns to numeric (safety)
numeric_cols = ["age", "heart_rate", "oxygen_level", "bp_systolic"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols)

# -----------------------------
# Risk Calculations
# -----------------------------
df["risk_level"] = df.apply(calculate_risk, axis=1)
df["risk_reason"] = df.apply(explain_risk, axis=1)
df["NEWS_score"] = df.apply(calculate_news, axis=1)

# -----------------------------
# Dashboard Metrics
# -----------------------------
st.subheader("📊 Patient Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", len(df))
col2.metric("High Risk", (df["risk_level"] == "High").sum())
col3.metric("Medium Risk", (df["risk_level"] == "Medium").sum())
col4.metric("Avg NEWS Score", round(df["NEWS_score"].mean(), 2))

# -----------------------------
# Alerts
# -----------------------------
high_risk_df = df[df["risk_level"] == "High"]

if not high_risk_df.empty:
    st.error(f"🚨 ALERT: {len(high_risk_df)} High-Risk Patients Detected!")
else:
    st.success("✅ No High-Risk Patients Detected")

# -----------------------------
# Data Table
# -----------------------------
st.subheader("📋 Patient Risk Table")

MAX_ROWS_FOR_STYLE = 500
if len(df) <= MAX_ROWS_FOR_STYLE:
    st.dataframe(df.style.apply(highlight_risk, axis=1))
else:
    st.dataframe(df)

# -----------------------------
# Download Section
# -----------------------------
st.subheader("⬇ Download Reports")

st.download_button(
    "Download High-Risk Patients",
    high_risk_df.to_csv(index=False),
    file_name="high_risk_patients.csv",
    mime="text/csv"
)

# -----------------------------
# Feature Info
# -----------------------------
with st.expander("ℹ Advanced Features Implemented"):
    st.markdown("""
- ✅ Rule-based patient deterioration detection  
- ✅ ICU-specific NEWS scoring  
- ✅ Explainable risk reasons  
- ✅ Alert system for critical patients  
- ✅ Streamlit Cloud deployment  

**Future Enhancements:**
- Live sensor streaming
- LSTM-based time-series prediction
- Doctor login & patient history
""")
