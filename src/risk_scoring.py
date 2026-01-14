import pandas as pd

def _get_value(row, col):
    """
    Safely extract a scalar value whether row is:
    - pandas Series
    - 1-row pandas DataFrame
    """
    if isinstance(row, pd.DataFrame):
        return row.iloc[0][col] if col in row.columns else None
    elif isinstance(row, pd.Series):
        return row[col] if col in row.index else None
    return None


def compute_news(row):
    score = 0

    htn = _get_value(row, 'preop_htn')
    hb = _get_value(row, 'preop_hb')
    bmi = _get_value(row, 'bmi')
    cr = _get_value(row, 'preop_cr')

    if htn == 1:
        score += 1

    if hb is not None and hb < 10:
        score += 1

    if bmi is not None and (bmi < 18.5 or bmi > 30):
        score += 1

    if cr is not None and cr > 1.5:
        score += 1

    return score


def compute_mews(row):
    score = 0

    age = _get_value(row, 'age')
    gluc = _get_value(row, 'preop_gluc')
    ebl = _get_value(row, 'intraop_ebl')

    if age is not None and age > 65:
        score += 1

    if gluc is not None and gluc > 180:
        score += 1

    if ebl is not None and ebl > 500:
        score += 1

    return score
