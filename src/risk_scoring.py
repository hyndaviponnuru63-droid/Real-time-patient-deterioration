def compute_news(row):
    """
    Simplified NEWS scoring example:
    - vitals missing, assume 0
    """
    score = 0
    # Example scoring logic
    if 'preop_htn' in row and row['preop_htn'] == 1:
        score += 1
    if 'preop_dm' in row and row['preop_dm'] == 1:
        score += 1
    if 'bmi' in row:
        if row['bmi'] > 30:
            score += 1
        elif row['bmi'] < 18.5:
            score += 2
    return score

def compute_mews(row):
    score = 0
    if 'preop_htn' in row and row['preop_htn'] == 1:
        score += 1
    if 'preop_hb' in row and row['preop_hb'] < 10:
        score += 2
    return score
