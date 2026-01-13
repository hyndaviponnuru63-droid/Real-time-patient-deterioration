import pandas as pd
import numpy as np

def load_data(file_path="clinical_data.csv"):
    """Load CSV safely and handle empty/missing file"""
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please add your dataset.")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"{file_path} is empty. Please check your dataset.")
    return df

def preprocess_for_ml(df):
    """
    Select numeric columns only for ML, convert to float, fill missing
    """
    numeric_cols = [
        'age', 'bmi', 'asa', 'preop_htn', 'preop_dm',
        'preop_hb', 'preop_cr', 'preop_gluc', 
        'intraop_ebl', 'intraop_uo', 'intraop_rbc',
        'icu_days', 'death_inhosp'
    ]
    # Keep only columns present in dataset
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    df_ml = df[numeric_cols].copy()
    # Convert all to float
    df_ml = df_ml.astype(float)
    # Fill missing values
    df_ml = df_ml.fillna(df_ml.median())
    return df_ml
