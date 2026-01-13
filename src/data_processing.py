import pandas as pd
import numpy as np
import os

def load_data(file_path="clinical_data.csv"):
    """
    Safely load CSV file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please place your dataset in project root.")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"{file_path} is empty. Please check the dataset.")

    return df


def preprocess_for_ml(df):
    """
    FINAL FIX:
    - Select only numeric columns
    - Safely convert strings → numbers
    - Replace invalid values with median
    """

    numeric_cols = [
        'age', 'bmi', 'asa',
        'preop_htn', 'preop_dm',
        'preop_hb', 'preop_cr', 'preop_gluc',
        'intraop_ebl', 'intraop_uo', 'intraop_rbc',
        'icu_days', 'death_inhosp'
    ]

    # Keep only columns that exist
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    df_ml = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Fill NaN values caused by strings / empty cells
    df_ml = df_ml.fillna(df_ml.median())

    return df_ml
