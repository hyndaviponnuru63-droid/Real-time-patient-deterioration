import streamlit as st
import pandas as pd
from src.data_processing import load_data, preprocess_for_ml
from src.lstm_model import train_lstm, predict_lstm
from src.live_sensor import simulate_live_sensor
from src.alerts import generate_risk_summary

st.set_page_config(page_title="ICU Dashboard", layout="wide")
st.title("🫀 Real-Time ICU Patient Deterioration Monitor")

# ---------------- Load and preprocess ----------------
@st.cache_data
def get_data():
    df = load_data("clinical_data.csv")
    df_ml = preprocess_for_ml(df)
    return df, df_ml

df, df_ml = get_data()

# ---------------- Train LSTM model ----------------
@st.cache_resource
def get_model():
    model, scaler, feature_cols = train_lstm(df_ml)
    return model, scaler, feature_cols

model, scaler, feature_cols = get_model()

# ---------------- Select patient ----------------
patient_ids = df["subjectid"].unique()
selected_patient = st.selectbox("Select Patient ID", patient_ids)

patient_rows = df[df["subjectid"] == selected_patient]
if patient_rows.empty:
    st.error(f"No data found for patient ID: {selected_patient}")
    st.stop()

patient_row = patient_rows.iloc[0]
st.markdown(f"### 🧑‍⚕️ Currently Monitoring Patient ID: {selected_patient}")

# ---------------- Session state ----------------
if "last_patient" not in st.session_state or st.session_state.last_patient != selected_patient:
    st.session_state.last_patient = selected_patient
    st.session_state.risk_history = []

# ---------------- Live sensor ----------------
sensor = simulate_live_sensor(patient_row)
status_box = st.empty()
trend_box = st.empty()
data_box = st.empty()

# ---------------- Live monitoring loop ----------------
for live_df in sensor:
    if live_df.empty:
        continue  # skip if no live data

    # Predict risk
    ml_risk = predict_lstm(model, scaler, feature_cols, live_df)
    st.session_state.risk_history.append(ml_risk)

    # Keep only last 20 points for trend
    if len(st.session_state.risk_history) > 20:
        st.session_state.risk_history = st.session_state.risk_history[-20:]

    # Generate status and reasons
    status, reasons = generate_risk_summary(
        live_df.iloc[0],  # latest patient row
        ml_risk,
        st.session_state.risk_history
    )

    # ---------------- Display status ----------------
    if status == "CRITICAL":
        status_box.error("🔴 CRITICAL CONDITION")
    elif status == "MONITOR":
        status_box.warning("🟡 NEEDS MONITORING")
    else:
        status_box.success("🟢 Patient stable. No warning signs.")

    # ---------------- Display trend ----------------
    trend_data = pd.DataFrame(st.session_state.risk_history, columns=["Risk Score"])
    trend_box.line_chart(trend_data)

    # ---------------- Display live data ----------------
    data_box.dataframe(live_df)

