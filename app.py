import streamlit as st
import pandas as pd

from src.risk_scoring import (
    calculate_risk,
    explain_risk,
    calculate_news,
    highlight_risk
)

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Patient Deterioration Alert System",
    layout="wide"
)

st.title("🩺 Patient Deterioration Alert System")

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("clinical_data.csv")
except Exception as e:
    st.error(" Unable to load clinical_data.csv")
    st.stop()

# -----------------------------
# Standardize Column Names
# -----------------------------
df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# -----------------------------
# Auto Column Alias Mapping
# -----------------------------
column_aliases = {
    "heartrate": "heart_rate",
    "heart_rat": "heart_rate",
    "hr": "heart_rate",

    "spo2": "oxygen_level",
    "o2": "oxygen_level",
    "oxygen": "oxygen_level",

    "bp": "bp_systolic",
    "systolic_bp": "bp_systolic",
    "sbp": "bp_systolic"
}

df.rename(columns=column_aliases, inplace=True)

# -----------------------------
# Required Columns Check
# -----------------------------
required_columns = [
    "age",
    "heart_rate",
    "oxygen_level",
    "bp_systolic"
]

missing_cols = [c for c in required_columns if c not in df.columns]

if missing_cols:
    st.error(f" Missing required columns: {missing_cols}")
    st.stop()

# -----------------------------
# Convert to Numeric (Fix TypeError)
# -----------------------------
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)

# -----------------------------
# Risk Calculations
# -----------------------------
df["risk_level"] = df.apply(calculate_risk, axis=1)
df["risk_reason"] = df.apply(explain_risk, axis=1)
df["news_score"] = df.apply(calculate_news, axis=1)

# -----------------------------
# Dashboard Metrics
# -----------------------------
st.subheader("📊 Patient Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Patients", len(df))
c2.metric("High Risk", (df["risk_level"] == "High").sum())
c3.metric("Medium Risk", (df["risk_level"] == "Medium").sum())
c4.metric("Avg NEWS Score", round(df["news_score"].mean(), 2))

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
st.subheader("⬇ Download Report")

st.download_button(
    label="Download High-Risk Patients",
    data=high_risk_df.to_csv(index=False),
    file_name="high_risk_patients.csv",
    mime="text/csv"
)

# -----------------------------
# Feature Summary
# -----------------------------
with st.expander("ℹ System Features"):
    st.markdown("""
- ✅ Automatic column detection & correction  
- ✅ Rule-based deterioration scoring  
- ✅ ICU-specific NEWS scoring  
- ✅ Explainable risk reasons  
- ✅ Real-time alert banner  
- ✅ Downloadable clinical reports  
- ✅ Streamlit Cloud deployment ready  

**Planned Enhancements**
- Live sensor simulation  
- Time-series ML (LSTM)  
- Doctor login & patient history  
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed as a healthcare decision-support system for early patient deterioration detection.")
