import streamlit as st
import pandas as pd
import io
from src.data_processing import load_data, preprocess_for_ml
from src.lstm_model import train_lstm, predict_lstm
from src.live_sensor import simulate_live_sensor
from src.alerts import generate_risk_summary

st.set_page_config(page_title="ICU Dashboard", layout="wide")
st.title("🫀 Real-Time ICU Patient Deterioration Monitor")

# ------------------- Load & preprocess data -------------------
try:
    df = load_data("clinical_data.csv")
except FileNotFoundError:
    st.error("❌ clinical_data.csv not found! Upload the file and rerun.")
    st.stop()

df_ml = preprocess_for_ml(df)

# ------------------- Train ML model -------------------
model, scaler, feature_cols = train_lstm(df_ml)

# ------------------- High-Risk Patients Table -------------------
st.markdown("## 🚨 High-Risk Patients Overview")

risk_summary_list = []
for _, patient_row in df.iterrows():
    ml_risk = predict_lstm(model, scaler, feature_cols, patient_row.to_frame().T)
    status, reasons = generate_risk_summary(patient_row, ml_risk, [])
    if status in ["CRITICAL", "MONITOR"]:
        risk_summary_list.append({
            "subjectid": patient_row["subjectid"],
            "status": status,
            "ML Risk": round(ml_risk, 2),
            "Reasons": "; ".join(reasons)
        })

if risk_summary_list:
    risk_df = pd.DataFrame(risk_summary_list)

    # Color-coded table
    def color_status(val):
        if val == "CRITICAL":
            return "background-color: red; color: white"
        elif val == "MONITOR":
            return "background-color: yellow; color: black"
        else:
            return ""

    MAX_ROWS_FOR_STYLE = 500
    if len(risk_df) <= MAX_ROWS_FOR_STYLE:
        st.dataframe(risk_df.style.applymap(color_status, subset=["status"]))
    else:
        st.warning(f"Large dataset ({len(risk_df)} rows). Showing table without colors.")
        st.dataframe(risk_df)

    # CSV download
    csv_buffer = io.StringIO()
    risk_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download High-Risk Patients CSV",
        data=csv_buffer.getvalue(),
        file_name="high_risk_patients.csv",
        mime="text/csv"
    )
else:
    st.info("No CRITICAL or MONITOR patients at the moment.")

# ------------------- Single Patient Monitoring -------------------
st.markdown("## 🧑‍⚕️ Single Patient Live Monitoring")

patient_ids = df["subjectid"].unique()
selected_patient = st.selectbox("Select Patient ID", patient_ids)
patient_row = df[df["subjectid"] == selected_patient].iloc[0]

# Session state for risk history
if "last_patient" not in st.session_state:
    st.session_state.last_patient = selected_patient
    st.session_state.risk_history = []

if selected_patient != st.session_state.last_patient:
    st.session_state.risk_history = []
    st.session_state.last_patient = selected_patient

# Placeholder boxes
status_box = st.empty()
trend_box = st.empty()
data_box = st.empty()

# Show initial content immediately
status_box.info("Initializing live monitoring…")
trend_box.info("Trend chart will appear here")
data_box.info("Patient vitals will appear here")

# ------------------- Live Sensor Loop -------------------
sensor = simulate_live_sensor(patient_row)

# Using Streamlit button to start live monitoring
if st.button("Start Live Monitoring"):
    for live_df in sensor:
        ml_risk = predict_lstm(model, scaler, feature_cols, live_df)
        st.session_state.risk_history.append(ml_risk)
        if len(st.session_state.risk_history) > 10:
            st.session_state.risk_history.pop(0)

        status, reasons = generate_risk_summary(
            live_df.iloc[0],
            ml_risk,
            st.session_state.risk_history
        )

        # Update status
        if status == "CRITICAL":
            status_box.error(f"🔴 CRITICAL CONDITION: {', '.join(reasons)}")
        elif status == "MONITOR":
            status_box.warning(f"🟡 NEEDS MONITORING: {', '.join(reasons)}")
        else:
            status_box.success(f"🟢 Stable: {', '.join(reasons)}")

        # Update trend
        if len(st.session_state.risk_history) >= 2:
            trend_data = pd.DataFrame(
                st.session_state.risk_history,
                columns=["Risk Score"]
            )
            trend_box.line_chart(trend_data)
        else:
            trend_box.info("Collecting live risk data for this patient…")

        # Update vitals table
        data_box.dataframe(live_df)
