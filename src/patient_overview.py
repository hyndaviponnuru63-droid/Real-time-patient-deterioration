import pandas as pd

def generate_patient_risk_table(df, df_ml, model, scaler, feature_cols, predict_fn):
    """
    Generates a full patient overview table with all relevant features
    """
    rows = []

    for idx in df.index:
        row_raw = df.loc[idx]
        row_ml = df_ml.loc[idx, feature_cols]

        row_ml_df = pd.DataFrame([row_ml])
        ml_risk = predict_fn(model, scaler, feature_cols, row_ml_df)

        # Determine status
        if ml_risk > 0.7:
            status = "CRITICAL"
        elif ml_risk > 0.4:
            status = "MONITOR"
        else:
            status = "STABLE"

        # Include all relevant features for display
        display_row = row_raw.to_dict()
        display_row.update({
            "ML_Risk": round(ml_risk, 3),
            "Status": status
        })

        rows.append(display_row)

    return pd.DataFrame(rows)
