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

for live_row in simulate_live_sensor(df_ml):
    live_df = pd.DataFrame([live_row])
    lstm_risk = predict_lstm(model, scaler, live_df.drop(['death_inhosp'], axis=1))
    alerts = check_alerts(live_row, lstm_risk=lstm_risk)
    
    live_df['LSTM_Risk'] = lstm_risk
    live_df['NEWS'] = live_df.apply(lambda x: check_alerts(x)[0] if len(check_alerts(x))>0 else 0, axis=1)
    
    placeholder.dataframe(live_df)
    if alerts:
        alerts_placeholder.warning(alerts)

# Download processed data
st.download_button(
    "Download processed data",
    df.to_csv(index=False),
    file_name="processed_icu_data.csv",
    mime="text/csv"
)
