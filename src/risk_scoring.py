def compute_news(row_df):
    """
    row_df: pandas DataFrame with exactly ONE row
    """
    row = row_df.iloc[0]  # convert to Series

    score = 0

    # Hypertension
    if 'preop_htn' in row and row['preop_htn'] == 1:
        score += 1

    # Low hemoglobin
    if 'preop_hb' in row and row['preop_hb'] < 10:
        score += 1

    # Abnormal BMI
    if 'bmi' in row and (row['bmi'] < 18.5 or row['bmi'] > 30):
        score += 1

    # Renal risk
    if 'preop_cr' in row and row['preop_cr'] > 1.5:
        score += 1

    return score


def compute_mews(row_df):
    """
    row_df: pandas DataFrame with exactly ONE row
    """
    row = row_df.iloc[0]  # convert to Series

    score = 0

    if 'age' in row and row['age'] > 65:
        score += 1

    if 'preop_gluc' in row and row['preop_gluc'] > 180:
        score += 1

    if 'intraop_ebl' in row and row['intraop_ebl'] > 500:
        score += 1

    return score
