import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_for_ml(df):
    # select only numeric columns
    feature_cols = df.select_dtypes(include="number").columns.tolist()

    # target column
    if "death_inhosp" in feature_cols:
        feature_cols.remove("death_inhosp")

    # ML dataframe
    df_ml = df[feature_cols + ["death_inhosp"]].copy()
    df_ml = df_ml.fillna(df_ml.median(numeric_only=True))

    return df_ml, feature_cols


