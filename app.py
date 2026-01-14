import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(page_title="Real-time Patient Deterioration Alert System", layout="wide")
st.title("🚨 Real-time Patient Deterioration Alert System")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/clinical_data.csv", parse_dates=['casestart', 'caseend', 'anestart', 'aneend', 'opstart', 'opend'])
    df = df.fillna(0)  # handle missing values simply
    return df

df = load_data()
st.success(f"✅ Data loaded successfully. Dataset shape: {df.shape}")

# -----------------------------
# Sidebar: Patient selector
# -----------------------------
patient_ids = df['subjectid'].unique()
selected_patient = st.sidebar.selectbox("Select Patient ID", patient_ids)

# Filter dataset for selected patient
patient_data = df[df['subjectid'] == selected_patient].copy()

# -----------------------------
# Risk Scoring Function
# -----------------------------
def calculate_risk(row):
    """Simple example: mark high-risk patients"""
    # Example: high risk if ICU days > 5 or death_inhosp = 1
    if row['icu_days'] > 5 or row['death_inhosp'] == 1:
        return "High"
    else:
        return "Low"

df['risk'] = df.apply(calculate_risk, axis=1)

# -----------------------------
# Display High-Risk Patients
# -----------------------------
high_risk_df = df[df['risk'] == "High"]
st.subheader("🚨 High-Risk Patient Overview")
st.dataframe(high_risk_df.head(5))  # Show 5 rows initially

# Download CSV button
st.download_button(
    label="📥 Download High-Risk Patients CSV",
    data=high_risk_df.to_csv(index=False),
    file_name="high_risk_patients.csv",
    mime="text/csv"
)

# -----------------------------
# Live Trend Simulation (Vitals)
# -----------------------------
st.subheader("📈 Patient Live Trends")

# Example vitals (replace with your dataset columns)
vitals_columns = ['heart_rate', 'oxygen_level', 'bp_systolic']

# Check if vitals columns exist
for col in vitals_columns:
    if col not in patient_data.columns:
        patient_data[col] = np.random.randint(60, 120, size=len(patient_data))  # simulate if missing

# Plot trend for selected patient
fig = px.line(patient_data, x='casestart', y=vitals_columns,
              labels={'value': 'Vital Signs', 'casestart': 'Time', 'variable': 'Vitals'},
              title=f"Live Trend for Patient ID: {selected_patient}")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Status Updates Simulation
# -----------------------------
st.subheader("🔔 Status Updates")
status_placeholder = st.empty()

# Simulate real-time status updates
statuses = [
    "Patient stable",
    "Slight deterioration observed",
    "High-risk alert triggered",
    "Vitals returning to normal",
    "Critical condition! Immediate attention required"
]

# Display updates for selected patient
for i in range(3):  # simulate 3 status updates
    status = np.random.choice(statuses)
    status_placeholder.info(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
    time.sleep(1)

# -----------------------------
# End of App
# -----------------------------
st.success("✅ Dashboard loaded successfully")
