import streamlit as st
import pandas as pd

from src.data_processing import load_data, preprocess_for_ml
from src.lstm_model import train_lstm, predict_lstm
from src.live_sensor import simulate_live_sensor
from src.alerts import generate_risk_summary

st.set_page_config(page_title="ICU Dashboard", layout="wide")
st.title("Real-Time ICU Patient Deterioration Monitor")

# -----------------------------
# Load & preprocess data
# -----------------------------
df = load_data("clinical_data.csv")
df_ml = preprocess_for_ml(df)

# -----------------------------
# Train LSTM model
# -----------------------------
model, scaler, feature_cols = train_lstm(df_ml)

# =====================================================
# GLOBAL RISK CLASSIFICATION (ALL PATIENTS)
# =====================================================
critical_rows = []
monitor_rows = []

for _, row in df.iterrows():
    row_df = pd.DataFrame([row])

    ml_risk = predict_lstm(model, scaler, feature_cols, row_df)
    status, reasons = generate_risk_summary(row, ml_risk, [])

    record = {
        "subjectid": row["subjectid"],
        "age": row.get("age"),
        "sex": row.get("sex"),
        "NEWS_reason": ", ".join(reasons),
        "ML_risk": round(ml_risk, 3),
        "status": status
    }

    if status == "CRITICAL":
        critical_rows.append(record)
    elif status == "MONITOR":
        monitor_rows.append(record)

critical_df = pd.DataFrame(critical_rows)
monitor_df = pd.DataFrame(monitor_rows)

# =====================================================
# SIDEBAR: DOWNLOAD SECTION
# =====================================================
st.sidebar.header("Patient Risk Lists")

if not critical_df.empty:
    st.sidebar.download_button(
        "Download CRITICAL patients (CSV)",
        critical_df.to_csv(index=False),
        "critical_patients.csv",
        "text/csv"
    )

if not monitor_df.empty:
    st.sidebar.download_button(
        "Download MONITOR patients (CSV)",
        monitor_df.to_csv(index=False),
        "monitor_patients.csv",
        "text/csv"
    )

# =====================================================
# DISPLAY TABLES
# =====================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Critical Patients")
    if critical_df.empty:
        st.success("No critical patients detected")
    else:
        st.dataframe(critical_df)

with col2:
    st.subheader("Monitor Patients")
    if monitor_df.empty:
        st.success("No monitor patients detected")
    else:
        st.dataframe(monitor_df)

# =====================================================
# LIVE SINGLE-PATIENT MONITORING
# =====================================================
st.divider()
st.subheader("Live Patient Monitoring")

patient_ids = df["subjectid"].unique()
selected_patient = st.selectbox("Select Patient ID", patient_ids)

patient_row = df[df["subjectid"] == selected_patient].iloc[0]

if "last_patient" not in st.session_state:
    st.session_state.last_patient = selected_patient
    st.session_state.risk_history = []

if selected_patient != st.session_state.last_patient:
    st.session_state.risk_history = []
    st.session_state.last_patient = selected_patient

sensor = simulate_live_sensor(patient_row)

status_box = st.empty()
trend_box = st.empty()
data_box = st.empty()

for live_df in sensor:
    ml_risk = predict_lstm(model, scaler, feature_cols, live_df)

    if ml_risk is not None:
        st.session_state.risk_history.append(float(ml_risk))
    if len(st.session_state.risk_history) > 10:
        st.session_state.risk_history.pop(0)

    status, reasons = generate_risk_summary(
        live_df.iloc[0],
        ml_risk,
        st.session_state.risk_history
    )

    if status == "CRITICAL":
        status_box.error("CRITICAL CONDITION")
    elif status == "MONITOR":
        status_box.warning("NEEDS MONITORING")
    else:
        status_box.success("PATIENT STABLE")

    if len(st.session_state.risk_history) >= 2:
        trend_box.line_chart(
            pd.DataFrame(st.session_state.risk_history, columns=["Risk Score"])
        )
    else:
        trend_box.info("Collecting live risk data...")

    data_box.dataframe(live_df)
