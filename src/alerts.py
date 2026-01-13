import streamlit as st

def check_alerts(row):
    alerts = []
    from src.risk_scoring import compute_news, compute_mews
    news = compute_news(row)
    mews = compute_mews(row)
    if news >= 3:
        alerts.append(f"High NEWS score: {news}")
    if mews >= 3:
        alerts.append(f"High MEWS score: {mews}")
    return alerts
