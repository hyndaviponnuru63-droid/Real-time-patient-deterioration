import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_for_ml(df):
    # Select numerical features only and fill NaNs
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    df_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
    return df_numeric
