from src.risk_scoring import compute_news, compute_mews

def check_alerts(row, lstm_risk=None):
    alerts = []
    news = compute_news(row)
    mews = compute_mews(row)
    if news >= 3:
        alerts.append(f"High NEWS score: {news}")
    if mews >= 3:
        alerts.append(f"High MEWS score: {mews}")
    if lstm_risk is not None and lstm_risk > 0.5:
        alerts.append(f"High ML Risk: {lstm_risk:.2f}")
    return alerts
