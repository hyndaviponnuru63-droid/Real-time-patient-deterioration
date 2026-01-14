import streamlit as st
import pandas as pd
from src.data_processing import load_data, preprocess_for_ml
from src.lstm_model import train_lstm, predict_lstm
from src.live_sensor import simulate_live_sensor
from src.alerts import generate_risk_summary

st.set_page_config(page_title="ICU Dashboard", layout="wide")
st.title("Real-Time ICU Patient Deterioration Monitor")

# ---------------------------------
# Load & Train
# ---------------------------------
df = load_data("clinical_data.csv")
df_ml = preprocess_for_ml(df)

model, scaler, feature_cols = train_lstm(df_ml)

# =================================
# GLOBAL PATIENT RISK TABLES
# =================================
critical, monitor = [], []

for _, row in df.iterrows():
    row_df = pd.DataFrame([row])
    ml_risk = predict_lstm(model, scaler, feature_cols, row_df)
    status, reasons = generate_risk_summary(row, ml_risk, [])

    record = {
        "subjectid": row["subjectid"],
        "age": row.get("age"),
        "sex": row.get("sex"),
        "ml_risk": round(ml_risk, 3),
        "reasons": ", ".join(reasons),
        "status": status
    }

    if status == "CRITICAL":
        critical.append(record)
    elif status == "MONITOR":
        monitor.append(record)

critical_df = pd.DataFrame(critical)
monitor_df = pd.DataFrame(monitor)

# =================================
# DISPLAY + DOWNLOAD
# =================================
st.subheader("Patient Risk Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Critical Patients")
    if not critical_df.empty:
        st.dataframe(critical_df)
        st.download_button(
            "Download Critical CSV",
            critical_df.to_csv(index=False),
            "critical_patients.csv",
            "text/csv"
        )
    else:
        st.success("No critical patients")

with col2:
    st.markdown("### Monitor Patients")
    if not monitor_df.empty:
        st.dataframe(monitor_df)
        st.download_button(
            "Download Monitor CSV",
            monitor_df.to_csv(index=False),
            "monitor_patients.csv",
            "text/csv"
        )
    else:
        st.success("No patients need monitoring")

# =================================
# LIVE MONITORING SECTION
# =================================
st.divider()
st.subheader("Live Patient Monitoring")

patient_ids = df["subjectid"].unique()
selected_patient = st.selectbox("Select Patient ID", patient_ids)

start_live = st.button("Start Live Monitoring")

if start_live:
    patient_row = df[df["subjectid"] == selected_patient].iloc[0]
    sensor = simulate_live_sensor(patient_row)

    status_box = st.empty()
    trend_box = st.empty()
    table_box = st.empty()

    if "risk_history" not in st.session_state:
        st.session_state.risk_history = []

    for _ in range(20):  # ✅ LIMIT LOOP
        live_df = next(sensor)

        ml_risk = predict_lstm(model, scaler, feature_cols, live_df)
        st.session_state.risk_history.append(ml_risk)

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

        trend_box.line_chart(
            pd.DataFrame(st.session_state.risk_history, columns=["Risk"])
        )

        table_box.dataframe(live_df)

    st.info("Live monitoring cycle completed")
    st.stop()
