from src.risk_scoring import compute_news, compute_mews

def generate_risk_summary(row, ml_risk, history):
    news = compute_news(row)
    mews = compute_mews(row)

    if ml_risk > 0.7 or news + mews >= 2:
        return "CRITICAL", ["High ML risk"]
    elif ml_risk > 0.4:
        return "MONITOR", ["Moderate risk"]
    else:
        return "STABLE", []
