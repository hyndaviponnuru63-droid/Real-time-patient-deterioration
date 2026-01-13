import streamlit as st
import pandas as pd

from src.risk_scoring import (
    calculate_risk,
    explain_risk,
    calculate_news,
    highlight_risk
)

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Patient Deterioration Alert System",
    layout="wide"
)

st.title("🩺 Patient Deterioration Alert System")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
try:
    df = pd.read_csv("clinical_data.csv")
except Exception:
    st.error("❌ Unable to load clinical_data.csv")
    st.stop()

# -------------------------------------------------
# Standardize Column Names
# -------------------------------------------------
df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

# -------------------------------------------------
# SMART AUTO COLUMN DETECTION
# -------------------------------------------------
def find_column(keywords):
    for col in df.columns:
        for kw in keywords:
            if kw in col:
                return col
    return None

column_map = {
    "age": find_column(["age"]),
    "heart_rate": find_column(["heart", "pulse", "hr"]),
    "oxygen_level": find_column(["oxygen", "spo2", "o2"]),
    "bp_systolic": find_column(["bp", "systolic"])
}

# -------------------------------------------------
# Validate Required Columns
# -------------------------------------------------
if None in column_map.values():
    st.error("Unable to auto-detect required vital sign columns.")
    st.write("📌 Detected columns in dataset:")
    st.write(list(df.columns))
    st.stop()

# Rename detected columns to standard names
df = df.rename(columns={
    column_map["age"]: "age",
    column_map["heart_rate"]: "heart_rate",
    column_map["oxygen_level"]: "oxygen_level",
    column_map["bp_systolic"]: "bp_systolic"
})

# -------------------------------------------------
# Convert to Numeric (Fix Type Errors)
# -------------------------------------------------
required_columns = ["age", "heart_rate", "oxygen_level", "bp_systolic"]

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)

# -------------------------------------------------
# Risk Calculations
# -------------------------------------------------
df["risk_level"] = df.apply(calculate_risk, axis=1)
df["risk_reason"] = df.apply(explain_risk, axis=1)
df["news_score"] = df.apply(calculate_news, axis=1)

# -------------------------------------------------
# Dashboard Metrics
# -------------------------------------------------
st.subheader("📊 Patient Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Patients", len(df))
c2.metric("High Risk", (df["risk_level"] == "High").sum())
c3.metric("Medium Risk", (df["risk_level"] == "Medium").sum())
c4.metric("Avg NEWS Score", round(df["news_score"].mean(), 2))

# -------------------------------------------------
# Alerts
# -------------------------------------------------
high_risk_df = df[df["risk_level"] == "High"]

if not high_risk_df.empty:
    st.error(f"🚨 ALERT: {len(high_risk_df)} High-Risk Patients Detected!")
else:
    st.success("✅ No High-Risk Patients Detected")

# -------------------------------------------------
# Patient Risk Table
# -------------------------------------------------
st.subheader("📋 Patient Risk Table")

MAX_ROWS_FOR_STYLE = 500

if len(df) <= MAX_ROWS_FOR_STYLE:
    st.dataframe(df.style.apply(highlight_risk, axis=1))
else:
    st.dataframe(df)

# -------------------------------------------------
# Download Section
# -------------------------------------------------
st.subheader("⬇ Download Report")

st.download_button(
    label="Download High-Risk Patients",
    data=high_risk_df.to_csv(index=False),
    file_name="high_risk_patients.csv",
    mime="text/csv"
)

# -------------------------------------------------
# Feature Summary
# -------------------------------------------------
with st.expander("ℹ System Features"):
    st.markdown("""
- ✅ Smart auto-detection of clinical columns  
- ✅ Rule-based patient deterioration scoring  
- ✅ ICU-specific NEWS scoring  
- ✅ Explainable risk reasons  
- ✅ Real-time alert banner  
- ✅ Downloadable reports  
- ✅ Streamlit Cloud deployment ready  

**Future Enhancements**
- Live sensor simulation  
- Time-series ML (LSTM)  
- Doctor login & patient history  
""")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Clinical decision-support system for early patient deterioration detection.")
