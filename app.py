import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

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

# Map available features (vitals may be missing)
column_map = {
    "age": find_column(["age"]),
    "heart_rate": find_column(["heart", "pulse", "hr"]),
    "oxygen_level": find_column(["oxygen", "spo2", "o2"]),
    "bp_systolic": find_column(["bp", "systolic"])
}

# -------------------------------------------------
# Handle Missing Vitals with Simulation
# -------------------------------------------------
vital_missing = False
for key, col in column_map.items():
    if col is None:
        vital_missing = True
        # Simulate placeholder values for missing vitals
        if key == "heart_rate":
            df["heart_rate"] = np.random.randint(60, 100, size=len(df))
        elif key == "oxygen_level":
            df["oxygen_level"] = np.random.randint(92, 100, size=len(df))
        elif key == "bp_systolic":
            df["bp_systolic"] = np.random.randint(110, 140, size=len(df))
        elif key == "age":
            df["age"] = np.random.randint(20, 80, size=len(df))

# Rename detected columns to standard names
df = df.rename(columns={v: k for k, v in column_map.items() if v is not None})

if vital_missing:
    st.warning("⚠️ Some vitals were missing from dataset. Using simulated values.")

# -------------------------------------------------
# Convert to Numeric
# -------------------------------------------------
numeric_cols = ["age", "heart_rate", "oxygen_level", "bp_systolic"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=numeric_cols)

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
# Live Sensor Simulation (Vitals)
# -------------------------------------------------
st.subheader("💓 Live Sensor Simulation")

sensor_patient = st.selectbox("Select Patient (ID):", df["caseid"].tolist())

placeholder = st.empty()

def simulate_live_vitals(patient_id):
    # Fetch patient row
    row = df[df["caseid"] == patient_id].iloc[0]
    hr = row["heart_rate"]
    spo2 = row["oxygen_level"]
    bp = row["bp_systolic"]

    # Simulate small fluctuations
    hr_live = hr + np.random.randint(-5, 5)
    spo2_live = spo2 + np.random.randint(-2, 2)
    bp_live = bp + np.random.randint(-10, 10)
    
    placeholder.metric("Heart Rate (bpm)", hr_live)
    placeholder.metric("Oxygen Level (%)", spo2_live)
    placeholder.metric("Systolic BP (mmHg)", bp_live)

st.button("Update Vitals", on_click=simulate_live_vitals, args=(sensor_patient,))

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
# Doctor / Patient History Simulation
# -------------------------------------------------
st.subheader("🧾 Patient History")
patient_history_id = st.selectbox("Select Patient for History:", df["caseid"].tolist(), key="history")
st.write(f"Displaying historical data for Patient ID: {patient_history_id}")
st.table(df[df["caseid"] == patient_history_id].drop(columns=["risk_level", "risk_reason", "news_score"]))

# -------------------------------------------------
# Feature Summary & Future Enhancements
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
""")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Clinical decision-support system for early patient deterioration detection.")
