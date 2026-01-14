import streamlit as st
import pandas as pd
import numpy as np
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

# Display full risk table
st.dataframe(st.session_state.risk_table)

# Download full risk table
st.download_button(
    "⬇️ Download Full Risk Table",
    st.session_state.risk_table.to_csv(index=False),
    "risk_table.csv",
    "text/csv"
)

# Show critical / monitor patients only
critical_df = st.session_state.risk_table[
    st.session_state.risk_table["Status"].isin(["CRITICAL", "MONITOR"])
]
if not critical_df.empty:
    st.markdown("### 🔴 Critical / Monitor Patients")
    st.dataframe(critical_df)

    # Download critical patients
    st.download_button(
        "⬇️ Download Critical / Monitor Patients CSV",
        critical_df.to_csv(index=False),
        "critical_patients.csv",
        "text/csv"
    )

# ---------------- SELECT PATIENT ----------------
st.markdown("## 🧑‍⚕️ Monitor a Patient Live")

patient_id = st.selectbox(
    "Select Patient ID",
    df["subjectid"].unique()
)

# Initialize session state
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []
if "live_index" not in st.session_state:
    st.session_state.live_index = 0
if "patient_data" not in st.session_state or st.session_state.patient_data["subjectid"].iloc[0] != patient_id:
    st.session_state.patient_data = df[df["subjectid"] == patient_id].reset_index(drop=True)
    st.session_state.risk_history = []
    st.session_state.live_index = 0

patient_data = st.session_state.patient_data

# Placeholders for live monitoring
status_box = st.empty()
trend_box = st.empty()
data_box = st.empty()

# Button to simulate live update
if st.button("▶️ Next Live Update"):

    idx = st.session_state.live_index % len(patient_data)
    live_row = patient_data.iloc[[idx]]
    st.session_state.live_index += 1

    # Simulate vitals if missing
    for col in ["heart_rate", "oxygen_level", "bp_systolic"]:
        if col not in live_row.columns:
            live_row[col] = np.random.randint(60, 120, size=1)

    # Predict risk
    ml_risk = predict_lstm(model, scaler, feature_cols, live_row)
    st.session_state.risk_history.append(ml_risk)
    st.session_state.risk_history = st.session_state.risk_history[-20:]

    # Status & reasons
    status, reasons = generate_risk_summary(
        live_row.iloc[0],
        ml_risk,
        st.session_state.risk_history
    )

    # Display status
    if status == "CRITICAL":
        status_box.error(f"🔴 CRITICAL CONDITION\nReasons: {', '.join(reasons)}")
    elif status == "MONITOR":
        status_box.warning(f"🟡 NEEDS MONITORING\nReasons: {', '.join(reasons)}")
    else:
        status_box.success("🟢 Patient stable. No warning signs.")

    # Live risk trend
    trend_box.line_chart(pd.DataFrame(st.session_state.risk_history, columns=["Risk Score"]))

    # Show current patient row
    data_box.dataframe(live_row)
