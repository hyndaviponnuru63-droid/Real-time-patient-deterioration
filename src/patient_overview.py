import pandas as pd
from src.risk_scoring import compute_news, compute_mews

def generate_patient_risk_table(df, model, scaler, feature_cols, predict_fn):
    rows = []

    for _, row in df.iterrows():
        row_df = pd.DataFrame([row])

        ml_risk = predict_fn(model, scaler, feature_cols, row_df)
        news = compute_news(row)
        mews = compute_mews(row)

        if news >= 5 or ml_risk > 0.6:
            status = "CRITICAL"
        elif news >= 3 or ml_risk > 0.4:
            status = "MONITOR"
        else:
            status = "STABLE"

        rows.append({
            "subjectid": row["subjectid"],
            "age": row["age"],
            "department": row["department"],
            "NEWS": news,
            "MEWS": mews,
            "ML_Risk": round(ml_risk, 3),
            "Status": status
        })

    return pd.DataFrame(rows)
