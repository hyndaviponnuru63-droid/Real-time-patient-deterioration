from src.risk_scoring import compute_news, compute_mews
import numpy as np

def generate_risk_summary(row, ml_risk, risk_history):
    news = compute_news(row)
    mews = compute_mews(row)

    reasons = []

    # Trend logic
    rising_trend = False
    if len(risk_history) >= 5:
        recent = risk_history[-5:]
        if all(x < y for x, y in zip(recent, recent[1:])):
            rising_trend = True
            reasons.append("ML risk increasing continuously")

    if news >= 3:
        reasons.append(f"NEWS score high ({news})")
    if mews >= 3:
        reasons.append(f"MEWS score high ({mews})")
    if ml_risk > 0.6:
        reasons.append(f"ML deterioration risk high ({ml_risk:.2f})")

    # ICU STATUS
    if ml_risk > 0.7 or news >= 5 or rising_trend:
        status = "🔴 CRITICAL"
    elif ml_risk > 0.4 or news >= 3:
        status = "🟡 MONITOR"
    else:
        status = "🟢 STABLE"

    return {
        "status": status,
        "news": news,
        "mews": mews,
        "ml_risk": ml_risk,
        "trend": rising_trend,
        "reasons": reasons
    }
