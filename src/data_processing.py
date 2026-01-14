import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)


def preprocess_for_ml(df):
    df_ml = df.copy()

    # -----------------------------
    # 1. Replace ICU-style strings
    # -----------------------------
    def clean_value(x):
        if isinstance(x, str):
            x = x.strip()
            if x.startswith(">"):
                return float(x[1:])   # '>89' → 89.0
            if x.startswith("<"):
                return float(x[1:])   # '<60' → 60.0
        return x

    df_ml = df_ml.applymap(clean_value)

    # -----------------------------
    # 2. Keep only numeric columns
    # -----------------------------
    df_ml = df_ml.select_dtypes(include=[np.number])

    # -----------------------------
    # 3. Fill missing values safely
    # -----------------------------
    df_ml = df_ml.fillna(df_ml.median(numeric_only=True))

    return df_ml
