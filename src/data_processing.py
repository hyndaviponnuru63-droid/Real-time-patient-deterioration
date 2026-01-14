import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_for_ml(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    numeric_cols.remove("death_inhosp")

    df_ml = df[numeric_cols + ["death_inhosp"]].copy()
    df_ml = df_ml.fillna(df_ml.median())

    return df_ml, numeric_cols
