import streamlit as st
import pandas as pd
from src.data_processing import load_data, preprocess_for_ml
from src.risk_scoring import compute_news, compute_mews
from src.alerts import check_alerts
from src.lstm_model import train_lstm, predict_lstm
from src.utils import simulate_live_sensor

st.set_page_config(page_title="ICU Patient Deterioration Dashboard", layout="wide")
st.title("Real-time ICU Patient Deterioration Monitoring")

# Load data
df = load_data("data/clinical_data.csv")
st.sidebar.header("Data Overview")
st.sidebar.write(f"Total records: {len(df)}")
if st.sidebar.checkbox("Show raw data"):
    st.dataframe(df.head())

# Preprocess for ML
df_ml = preprocess_for_ml(df)
model, scaler = train_lstm(df_ml)

# Live simulation
st.header("Live ICU Simulation")
placeholder = st.empty()
alerts_placeholder = st.empty()

for live_row in simulate_live_sensor(df_ml):
    live_df = pd.DataFrame([live_row])
    # Compute risk scores
    live_df['NEWS'] = live_df.apply(compute_news, axis=1)
    live_df['MEWS'] = live_df.apply(compute_mews, axis=1)
    live_df['LSTM_Risk'] = live_df.drop(['death_inhosp'], axis=1).pipe(lambda x: predict_lstm(model, scaler, x))
    
    # Show live data
    placeholder.dataframe(live_df)
    
    # Check alerts
    alerts = check_alerts(live_row)
    if live_row['LSTM_Risk'] > 0.5:
        alerts.append(f"High ML Risk: {live_row['LSTM_Risk']:.2f}")
    if alerts:
        alerts_placeholder.warning(alerts)

# Download processed data
st.download_button(
    "Download processed data",
    df.to_csv(index=False),
    file_name="processed_icu_data.csv",
    mime="text/csv"
)
