import streamlit as st
import pandas as pd
import time
from src.data_processing import load_data, preprocess_for_ml
from src.lstm_model import train_lstm, predict_lstm
from src.patient_overview import generate_patient_risk_table
from src.live_sensor import simulate_live_sensor
from src.alerts import generate_risk_summary

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

# ---------------- HIGH-RISK PATIENT OVERVIEW ----------------
st.markdown("## 🚨 High-Risk Patient Overview")

risk_table = generate_patient_risk_table(
    df,
    df_ml,
    model,
    scaler,
    feature_cols,
    predict_lstm
)

# Show full table with features
st.dataframe(risk_table)

# Download CSV button
st.download_button(
    "⬇️ Download High-Risk Patients CSV",
    risk_table.to_csv(index=False),
    "high_risk_patients.csv",
    "text/csv"
)

# Filter high-risk patients for reference
high_risk_df = risk_table[risk_table["Status"].isin(["CRITICAL", "MONITOR"])]
if not high_risk_df.empty:
    st.markdown("### 🔴 High-Risk Patients")
    st.dataframe(high_risk_df)

# ---------------- SELECT PATIENT ----------------
st.markdown("## 🧑‍⚕️ Select Patient to Monitor Live")

patient_id = st.selectbox(
    "Select Patient ID",
    df["subjectid"].unique()
)

patient_row = df[df["subjectid"] == patient_id].iloc[0]

# ---------------- SESSION STATE ----------------
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []

# ---------------- LIVE SENSOR ----------------
sensor = simulate_live_sensor(patient_row)
status_box = st.empty()
trend_box = st.empty()
data_box = st.empty()

# ---------------- LIVE MONITORING SIMULATION ----------------
st.markdown("### 🔴 Live Monitoring (simulated)")

for live_df in sensor:

    # Predict ML risk
    ml_risk = predict_lstm(model, scaler, feature_cols, live_df)
    st.session_state.risk_history.append(ml_risk)
    st.session_state.risk_history = st.session_state.risk_history[-20:]  # keep last 20 points

    # Generate status & reasons
    status, reasons = generate_risk_summary(
        live_df.iloc[0],
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

    # Display live risk trend
    trend_box.line_chart(
        pd.DataFrame(st.session_state.risk_history, columns=["Risk Score"])
    )

    # Display live patient row
    data_box.dataframe(live_df)

    # Small delay to simulate live update
    time.sleep(1)  # adjust for faster/slower updates

# ---------------- END OF LIVE MONITOR ----------------
st.success("✅ Live monitoring complete")
