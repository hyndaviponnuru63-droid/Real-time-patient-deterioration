import pandas as pd
from src.risk_scoring import compute_news, compute_mews

def generate_patient_risk_table(df, df_ml, model, scaler, feature_cols, predict_fn):
    rows = []

    for idx in df.index:
        row_raw = df.loc[idx]
        row_ml = df_ml.loc[idx]

        row_ml_df = pd.DataFrame([row_ml])

        ml_risk = predict_fn(model, scaler, feature_cols, row_ml_df)

        news = compute_news(row_raw)
        mews = compute_mews(row_raw)

        if news >= 5 or ml_risk > 0.6:
            status = "CRITICAL"
        elif news >= 3 or ml_risk > 0.4:
            status = "MONITOR"
        else:
            status = "STABLE"

        rows.append({
            "subjectid": row_raw["subjectid"],
            "age": row_raw["age"],
            "department": row_raw["department"],
            "NEWS": news,
            "MEWS": mews,
            "ML_Risk": round(float(ml_risk), 3),
            "Status": status
        })

    return pd.DataFrame(rows)
