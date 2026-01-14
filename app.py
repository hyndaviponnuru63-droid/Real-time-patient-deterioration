import streamlit as st
import pandas as pd

from src.data_processing import load_data, preprocess_for_ml
from src.lstm_model import train_lstm, predict_lstm
from src.live_sensor import simulate_live_sensor
from src.alerts import generate_risk_summary

st.set_page_config(
    page_title="ICU Patient Deterioration Monitor",
    layout="wide"
)

st.title("🫀 Real-Time ICU Patient Deterioration Dashboard")

# ---------------- LOAD DATA ----------------
df = load_data("clinical_data.csv")
df_ml = preprocess_for_ml(df)

# ---------------- TRAIN MODEL ----------------
model, scaler = train_lstm(df_ml)

# ---------------- SELECT PATIENT ----------------
patient_ids = df["subjectid"].unique()
selected_patient = st.selectbox("Select Patient ID", patient_ids)

patient_row = df[df["subjectid"] == selected_patient].iloc[0]

st.markdown(
    f"### 🧑‍⚕️ Currently Monitoring Patient ID: **{selected_patient}**"
)

# ---------------- SESSION STATE ----------------
if "last_patient" not in st.session_state:
    st.session_state.last_patient = selected_patient
    st.session_state.risk_history = []

if selected_patient != st.session_state.last_patient:
    st.session_state.risk_history = []   # reset trend
    st.session_state.last_patient = selected_patient

# ---------------- LIVE SENSOR ----------------
sensor = simulate_live_sensor(patient_row)

status_box = st.empty()
trend_box = st.empty()
data_box = st.empty()

# ---------------- LIVE LOOP ----------------
for live_df in sensor:

    # ML prediction (IMPORTANT FIX)
    ml_risk = predict_lstm(
        model,
        scaler,
        live_df.select_dtypes(include="number")
    )

    st.session_state.risk_history.append(ml_risk)

    if len(st.session_state.risk_history) > 10:
        st.session_state.risk_history.pop(0)

    status, reasons = generate_risk_summary(
        live_df,
        ml_risk,
        st.session_state.risk_history
    )

    # -------- STATUS --------
    if status == "CRITICAL":
        status_box.error("🔴 CRITICAL CONDITION")
    elif status == "MONITOR":
        status_box.warning("🟡 NEEDS MONITORING")
    else:
        status_box.success("🟢 Patient stable. No warning signs.")

    # -------- TREND --------
    trend_box.line_chart(st.session_state.risk_history)

    # -------- TABLE --------
    data_box.dataframe(live_df)
