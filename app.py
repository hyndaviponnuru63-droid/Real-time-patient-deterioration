import streamlit as st
import pandas as pd

from src.data_processing import load_data, preprocess_for_ml
from src.lstm_model import train_lstm, predict_lstm
from src.live_sensor import simulate_live_sensor
from src.alerts import generate_risk_summary

# ==================================================
# STREAMLIT PAGE CONFIG
# ==================================================
st.set_page_config(page_title="ICU Dashboard", layout="wide")
st.title("🫀 Real-Time ICU Patient Deterioration Monitor")

# ==================================================
# LOAD DATA
# ==================================================
df = load_data("clinical_data.csv")
df_ml = preprocess_for_ml(df)

# ==================================================
# CACHE MODEL (RESOURCE-HEAVY)
# ==================================================
@st.cache_resource
def load_model(df_ml):
    return train_lstm(df_ml)

model, scaler, feature_cols = load_model(df_ml)

# ==================================================
# CACHE RISK TABLES FUNCTION
# ==================================================
@st.cache_data
def build_risk_tables(df):
    critical_list = []
    monitor_list = []

    for _, row in df.iterrows():
        row_df = pd.DataFrame([row])
        ml_risk = predict_lstm(model, scaler, feature_cols, row_df)
        status, reasons = generate_risk_summary(row, ml_risk, [])

        record = {
            "subjectid": row["subjectid"],
            "age": row.get("age"),
            "sex": row.get("sex"),
            "ml_risk": round(float(ml_risk), 3),
            "reasons": ", ".join(reasons),
            "status": status
        }

        if status == "CRITICAL":
            critical_list.append(record)
        elif status == "MONITOR":
            monitor_list.append(record)

    critical_df = pd.DataFrame(critical_list)
    monitor_df = pd.DataFrame(monitor_list)

    # Top 10 for downloads
    critical_top10 = critical_df.sort_values("ml_risk", ascending=False).head(10)
    monitor_top10 = monitor_df.sort_values("ml_risk", ascending=False).head(10)

    return critical_df, monitor_df, critical_top10, monitor_top10

# ==================================================
# BUTTON TO GENERATE HIGH-RISK TABLES
# ==================================================
st.subheader("🚨 High-Risk Patient Lists")

if st.button("🔍 Generate Risk Tables"):
    with st.spinner("Analyzing patient risks..."):
        # Limit large dataset to avoid freezing
        limited_df = df.head(200)
        critical_df, monitor_df, critical_top10, monitor_top10 = build_risk_tables(limited_df)

    col1, col2 = st.columns(2)

    # ---- Critical Patients Table ----
    with col1:
        st.markdown("### 🔴 Critical Patients (All)")
        if not critical_df.empty:
            st.dataframe(critical_df)
            st.download_button(
                "⬇️ Download Top 10 Critical CSV",
                critical_top10.to_csv(index=False),
                file_name="critical_patients_top10.csv",
                mime="text/csv"
            )
        else:
            st.success("No critical patients found")

    # ---- Monitor Patients Table ----
    with col2:
        st.markdown("### 🟡 Monitor Patients (All)")
        if not monitor_df.empty:
            st.dataframe(monitor_df)
            st.download_button(
                "⬇️ Download Top 10 Monitor CSV",
                monitor_top10.to_csv(index=False),
                file_name="monitor_patients_top10.csv",
                mime="text/csv"
            )
        else:
            st.success("No monitor patients found")

st.divider()

# ==================================================
# LIVE PATIENT MONITORING
# ==================================================
st.subheader("🧑‍⚕️ Live Patient Monitoring")

patient_ids = df["subjectid"].unique()
selected_patient = st.selectbox("Select Patient ID", patient_ids)
patient_row = df[df["subjectid"] == selected_patient].iloc[0]

if "risk_history" not in st.session_state:
    st.session_state.risk_history = []

start_monitoring = st.button("▶️ Start Live Monitoring")

if start_monitoring:
    sensor = simulate_live_sensor(patient_row)

    status_box = st.empty()
    trend_box = st.empty()
    data_box = st.empty()

    for _ in range(20):  # Safe loop to avoid freezing
        live_df = next(sensor)

        ml_risk = predict_lstm(model, scaler, feature_cols, live_df)
        st.session_state.risk_history.append(float(ml_risk))
        if len(st.session_state.risk_history) > 10:
            st.session_state.risk_history.pop(0)

        status, reasons = generate_risk_summary(
            live_df.iloc[0],
            ml_risk,
            st.session_state.risk_history
        )

        if status == "CRITICAL":
            status_box.error(f"🔴 CRITICAL CONDITION\nReasons: {', '.join(reasons)}")
        elif status == "MONITOR":
            status_box.warning(f"🟡 NEEDS MONITORING\nReasons: {', '.join(reasons)}")
        else:
            status_box.success("🟢 Patient stable")

        if len(st.session_state.risk_history) >= 2:
            trend_box.line_chart(
                pd.DataFrame(
                    st.session_state.risk_history,
                    columns=["Risk Score"]
                )
            )

        data_box.dataframe(live_df)
