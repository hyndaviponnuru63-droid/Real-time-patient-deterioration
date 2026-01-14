import streamlit as st
import pandas as pd
import io
from src.data_processing import load_data, preprocess_for_ml
from src.lstm_model import train_lstm, predict_lstm
from src.alerts import generate_risk_summary

st.set_page_config(page_title="Patient Overview", layout="wide")
st.title("🧑‍⚕️ ICU Patient Overview Dashboard")

# ------------------- Load & preprocess -------------------
df = load_data("clinical_data.csv")
df_ml = preprocess_for_ml(df)

# ------------------- Train ML model -------------------
model, scaler, feature_cols = train_lstm(df_ml)

# ------------------- Generate Patient Risk Table -------------------
def generate_patient_overview(df, df_ml, model, scaler, feature_cols):
    rows = []

    for idx in df.index:
        row_raw = df.loc[idx]
        row_ml = df_ml.loc[idx, feature_cols]

        row_ml_df = pd.DataFrame([row_ml])
        ml_risk = predict_lstm(model, scaler, feature_cols, row_ml_df)

        # Use NEWS/MEWS + ML risk to get status & reasons
        status, reasons = generate_risk_summary(row_raw, ml_risk, [])

        display_row = row_raw.to_dict()
        display_row.update({
            "ML_Risk": round(ml_risk, 3),
            "Status": status,
            "Reasons": "; ".join(reasons)
        })
        rows.append(display_row)

    return pd.DataFrame(rows)

overview_df = generate_patient_overview(df, df_ml, model, scaler, feature_cols)

# ------------------- Filter & Search -------------------
st.markdown("## 🔍 Search / Filter Patients")
status_filter = st.multiselect(
    "Filter by Status",
    options=["CRITICAL", "MONITOR", "STABLE"],
    default=["CRITICAL", "MONITOR", "STABLE"]
)
filtered_df = overview_df[overview_df["Status"].isin(status_filter)]

subject_filter = st.text_input("Search by Patient ID (leave blank for all)")
if subject_filter:
    filtered_df = filtered_df[filtered_df["subjectid"].astype(str).str.contains(subject_filter)]

# ------------------- Color-coded table -------------------
def color_status(val):
    if val == "CRITICAL":
        return "background-color: red; color: white"
    elif val == "MONITOR":
        return "background-color: yellow; color: black"
    else:
        return ""

MAX_ROWS_FOR_STYLE = 500
if len(filtered_df) <= MAX_ROWS_FOR_STYLE:
    st.dataframe(filtered_df.style.applymap(color_status, subset=["Status"]))
else:
    st.warning(f"Large dataset ({len(filtered_df)} rows). Showing table without colors.")
    st.dataframe(filtered_df)

# ------------------- Download CSV -------------------
csv_buffer = io.StringIO()
filtered_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download Patient Overview CSV",
    data=csv_buffer.getvalue(),
    file_name="patient_overview.csv",
    mime="text/csv"
)
