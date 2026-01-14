import streamlit as st
import pandas as pd
import numpy as np
from src.data_processing import load_data, preprocess_for_ml
from src.lstm_model import train_lstm, predict_lstm
from src.patient_overview import generate_patient_risk_table
from src.live_sensor import simulate_live_sensor
from src.alerts import generate_risk_summary
from datetime import datetime
import time

st.set_page_config(page_title="ICU Dashboard", layout="wide")
st.title("🫀 Real-Time ICU Patient Deterioration Monitor")

# ---------------- LOAD DATA ----------------
df = load_data("clinical_data.csv")
df_ml, feature_cols = preprocess_for_ml(df)

st.success("✅ Data loaded successfully")
st.write("Dataset shape:", df.shape)

# ---------------- TRAIN MODEL ----------------
model, scaler = train_lstm(df_ml, feature_cols)
st.success("✅ LSTM model trained")

# ---------------- HIGH-RISK PATIENT TABLE ----------------
st.markdown("## 🚨 High-Risk Patient Overview")

if "risk_table" not in st.session_state:
    st.session_state.risk_table = generate_patient_risk_table(
        df,
        df_ml,
        model,
        scaler,
        feature_cols,
        predict_lstm
    )

st.dataframe(st.session_state.risk_table)

# Download CSV
st.download_button(
    "⬇️ Download High-Risk Patients CSV",
    st.session_state.risk_table.to_csv(index=False),
    "high_risk_patients.csv",
    "text/csv"
)

# ---------------- SELECT PATIENT ----------------
st.markdown("## 🧑‍⚕️ Select Patient to Monitor Live")

patient_id = st.selectbox(
    "Select Patient ID",
    df["subjectid"].unique()
)

# Initialize session state for live monitoring
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []
if "live_index" not in st.session_state:
    st.session_state.live_index = 0
if "patient_data" not in st.session_state:
    st.session_state.patient_data = df[df["subjectid"] == patient_id].reset_index(drop=True)

patient_data = st.session_state.patient_data

# ---------------- PLACEHOLDERS ----------------
status_box = st.empty()
trend_box = st.empty()
data_box = st.empty()

# ---------------- SIMULATE LIVE SENSOR ----------------
# Append random data to simulate live sensor if needed
if st.button("Simulate Next Live Update"):

    # Get next row (loop if we reach the end)
    idx = st.session_state.live_index % len(patient_data)
    live_row = patient_data.iloc[[idx]]  # keep as DataFrame
    st.session_state.live_index += 1

    # Simulate vitals if missing
    for col in ["heart_rate", "oxygen_level", "bp_systolic"]:
        if col not in live_row.columns:
            live_row[col] = np.random.randint(60, 120, size=1)

    # Predict risk
    ml_risk = predict_lstm(model, scaler, feature_cols, live_row)
    st.session_state.risk_history.append(ml_risk)
    st.session_state.risk_history = st.session_state.risk_history[-20:]  # last 20

    # Status and reasons
    status, reasons = generate_risk_summary(
        live_row.iloc[0],
        ml_risk,
        st.session_state.risk_history
    )

    # Display patient status
    if status == "CRITICAL":
        status_box.error(f"🔴 CRITICAL CONDITION\nReasons: {', '.join(reasons)}")
    elif status == "MONITOR":
        status_box.warning(f"🟡 NEEDS MONITORING\nReasons: {', '.join(reasons)}")
    else:
        status_box.success("🟢 Patient stable. No warning signs.")

    # Live risk trend
    trend_box.line_chart(
        pd.DataFrame(st.session_state.risk_history, columns=["Risk Score"])
    )

    # Show current patient row
    data_box.dataframe(live_row)
