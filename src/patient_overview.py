import pandas as pd

def generate_patient_risk_table(
    df,
    df_ml,
    model,
    scaler,
    feature_cols,
    predict_fn
):
    rows = []

    for _, row in df.iterrows():
        row_df = pd.DataFrame([row])
        ml_risk = predict_fn(model, scaler, feature_cols, row_df)

        if ml_risk > 0.7:
            status = "CRITICAL"
        elif ml_risk > 0.4:
            status = "MONITOR"
        else:
            status = "STABLE"

        rows.append({
            "subjectid": row["subjectid"],
            "Risk Score": round(ml_risk, 3),
            "Status": status
        })

    return pd.DataFrame(rows)
