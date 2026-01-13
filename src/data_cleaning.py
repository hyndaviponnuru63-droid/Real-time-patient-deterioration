import pandas as pd

def clean_numeric_columns(df):
    numeric_cols = [
        "age", "asa", "icu_days", "emop",
        "preop_hb", "preop_na", "preop_gluc"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["age"] = df["age"].fillna(df["age"].median())
    df["asa"] = df["asa"].fillna(2)
    df["icu_days"] = df["icu_days"].fillna(0)
    df["emop"] = df["emop"].fillna(0)
    df["preop_hb"] = df["preop_hb"].fillna(12)
    df["preop_na"] = df["preop_na"].fillna(140)
    df["preop_gluc"] = df["preop_gluc"].fillna(120)

    return df
