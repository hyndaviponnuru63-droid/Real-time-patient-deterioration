import streamlit as st
from src.data_processing import load_data, preprocess_for_ml
from src.lstm_model import train_lstm, predict_lstm
from src.alerts import check_alerts
from src.live_sensor import simulate_live_sensor
import pandas as pd

st.set_page_config(page_title="ICU Patient Deterioration Dashboard", layout="wide")
st.title("Real-time ICU Patient Deterioration Monitoring")

# Load CSV safely
try:
    df = load_data("clinical_data.csv")
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.header("Data Overview")
st.sidebar.write(f"Total records: {len(df)}")
if st.sidebar.checkbox("Show raw data"):
    st.dataframe(df.head())

# Preprocess numeric columns for ML
df_ml = preprocess_for_ml(df)

# Train LSTM
model, scaler = train_lstm(df_ml)

# Live simulation
st.header("Live ICU Simulation")
placeholder = st.empty()
alerts_placeholder = st.empty()
risk_history = []
from src.alerts import generate_risk_summary

st.header("🫀 Live ICU Patient Monitoring")

status_box = st.empty()
reason_box = st.empty()
trend_box = st.empty()
data_box = st.empty()

risk_history = []

for live_row in simulate_live_sensor(df_ml):

    live_df = pd.DataFrame([live_row])

    ml_risk = predict_lstm(
        model,
        scaler,
        live_df.drop(['death_inhosp'], axis=1)
    )

    # Store last 10 ML risk values
    risk_history.append(ml_risk)
    if len(risk_history) > 10:
        risk_history.pop(0)

    # Generate explainable risk
    risk = generate_risk_summary(live_row, ml_risk, risk_history)

    # 🔴🟡🟢 ICU STATUS PANEL
    if "CRITICAL" in risk["status"]:
        status_box.error(risk["status"])
    elif "MONITOR" in risk["status"]:
        status_box.warning(risk["status"])
    else:
        status_box.success(risk["status"])

    # 🧠 EXPLANATION PANEL
    if risk["reasons"]:
        reason_box.info("Reasons:\n- " + "\n- ".join(risk["reasons"]))
    else:
        reason_box.info("Patient stable. No warning signs.")

    # 📈 RISK TREND CHART
    trend_box.line_chart(
        pd.DataFrame({"ML Risk": risk_history})
    )

    # 📋 LIVE DATA TABLE
    live_df["NEWS"] = risk["news"]
    live_df["MEWS"] = risk["mews"]
    live_df["ML_Risk"] = round(ml_risk, 3)

    data_box.dataframe(live_df)

    if alerts:
        alerts_placeholder.warning(alerts)

# Download processed data
st.download_button(
    "Download processed data",
    df.to_csv(index=False),
    file_name="processed_icu_data.csv",
    mime="text/csv"
)

