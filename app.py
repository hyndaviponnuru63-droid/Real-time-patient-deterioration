import streamlit as st
import pandas as pd

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
st.dataframe(df.head(5))

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

st.write("Risk table shape:", risk_table.shape)
st.dataframe(risk_table.head(10))

high_risk = risk_table[risk_table["Status"].isin(["CRITICAL", "MONITOR"])]

if not high_risk.empty:
    st.dataframe(high_risk)
    st.download_button(
        "⬇️ Download High-Risk Patients",
        high_risk.to_csv(index=False),
        "high_risk_patients.csv",
        "text/csv"
    )
else:
    st.success("✅ No high-risk patients found")

# ---------------- SELECT PATIENT ----------------
st.markdown("## 🧑‍⚕️ Live Patient Monitoring")

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

# ---------------- LIVE LOOP ----------------
for live_df in sensor:

    ml_risk = predict_lstm(
        model,
        scaler,
        feature_cols,
        live_df
    )

    st.session_state.risk_history.append(ml_risk)
    st.session_state.risk_history = st.session_state.risk_history[-20:]

    status, reasons = generate_risk_summary(
        live_df.iloc[0],
        ml_risk,
        st.session_state.risk_history
    )

    if status == "CRITICAL":
        status_box.error("🔴 CRITICAL CONDITION")
    elif status == "MONITOR":
        status_box.warning("🟡 NEEDS MONITORING")
    else:
        status_box.success("🟢 Patient stable. No warning signs.")

    trend_box.line_chart(
        pd.DataFrame(st.session_state.risk_history, columns=["Risk Score"])
    )

    data_box.dataframe(live_df)

    break  # ⬅️ IMPORTANT: prevents infinite loop in Streamlit
