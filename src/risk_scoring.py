def compute_news(row):
    score = 0
    if row.get("preop_htn", 0) == 1:
        score += 1
    return score

def compute_mews(row):
    score = 0
    if row.get("age", 0) > 65:
        score += 1
    return score
