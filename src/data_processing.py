import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def preprocess_for_ml(df):
    # Select important features for LSTM
    features = [
        'age', 'bmi', 'asa', 'preop_htn', 'preop_dm',
        'preop_hb', 'preop_cr', 'preop_gluc', 
        'intraop_ebl', 'intraop_uo', 'intraop_rbc',
        'icu_days', 'death_inhosp'
    ]
    df_ml = df[features]
    df_ml = df_ml.fillna(0)
    return df_ml
