from src.risk_scoring import compute_news, compute_mews

def generate_risk_summary(row, ml_risk, risk_history):
    news = compute_news(row)
    mews = compute_mews(row)

    reasons = []

    # Always append the risk reasons
    if news >= 3:
        reasons.append(f"NEWS score high ({news})")
    if mews >= 3:
        reasons.append(f"MEWS score high ({mews})")
    if ml_risk > 0.4:
        reasons.append(f"ML deterioration risk elevated ({ml_risk:.2f})")

    # ICU status
    if news >= 5 or ml_risk > 0.6:
        status = "CRITICAL"
    elif news >= 3 or ml_risk > 0.4:
        status = "MONITOR"
    else:
        status = "STABLE"
        reasons.append("All vitals within normal limits")  # <-- optional for STABLE

    return status, reasons
