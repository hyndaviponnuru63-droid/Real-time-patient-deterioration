import streamlit as st
import pandas as pd
import io
from src.data_processing import load_data, preprocess_for_ml
from src.lstm_model import train_lstm, predict_lstm
from src.live_sensor import simulate_live_sensor
from src.alerts import generate_risk_summary

st.set_page_config(page_title="ICU Dashboard", layout="wide")
st.title("🫀 Real-Time ICU Patient Deterioration Monitor")

# ------------------- Load and preprocess data -------------------
df = load_data("clinical_data.csv")
df_ml = preprocess_for_ml(df)

# ------------------- Train ML model -------------------
model, scaler, feature_cols = train_lstm(df_ml)

# ------------------- Live Risk Summary Table -------------------
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
    st.markdown("## 🚨 Patients Needing Attention")

    # Color-coded table
    def color_status(val):
        if val == "CRITICAL":
            return "background-color: red; color: white"
        elif val == "MONITOR":
            return "background-color: yellow; color: black"
        else:
            return ""

    st.dataframe(risk_df.style.applymap(color_status, subset=["status"]))

    # CSV download
    csv_buffer = io.StringIO()
    risk_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Risk Patients CSV",
        data=csv_buffer.getvalue(),
        file_name="critical_monitor_patients.csv",
        mime="text/csv"
    )
else:
    st.info("No CRITICAL or MONITOR patients at the moment.")

# ------------------- Single Patient Monitoring -------------------
patient_ids = df["subjectid"].unique()
selected_patient = st.selectbox("Select Patient ID for Live Monitoring", patient_ids)
patient_row = df[df["subjectid"] == selected_patient].iloc[0]

st.markdown(f"### 🧑‍⚕️ Currently Monitoring Patient ID: {selected_patient}")

# Session state for risk history
if "last_patient" not in st.session_state:
    st.session_state.last_patient = selected_patient
    st.session_state.risk_history = []

if selected_patient != st.session_state.last_patient:
    st.session_state.risk_history = []
    st.session_state.last_patient = selected_patient

# Live sensor simulation
sensor = simulate_live_sensor(patient_row)
status_box = st.empty()
trend_box = st.empty()
data_box = st.empty()

# Live loop
for live_df in sensor:
    ml_risk = predict_lstm(model, scaler, feature_cols, live_df)
    st.session_state.risk_history.append(ml_risk)
    if len(st.session_state.risk_history) > 10:
        st.session_state.risk_history.pop(0)

    status, reasons = generate_risk_summary(
        live_df.iloc[0],  # pass Series
        ml_risk,
        st.session_state.risk_history
    )

    # Status display
    if status == "CRITICAL":
        status_box.error("🔴 CRITICAL CONDITION")
    elif status == "MONITOR":
        status_box.warning("🟡 NEEDS MONITORING")
    else:
        status_box.success("🟢 Patient stable. No warning signs.")

    # Trend line
    if len(st.session_state.risk_history) >= 2:
        trend_data = pd.DataFrame(
            st.session_state.risk_history,
            columns=["Risk Score"]
        )
        trend_box.line_chart(trend_data)
    else:
        trend_box.info("Collecting live risk data for this patient…")

    # Current vitals table
    data_box.dataframe(live_df)
