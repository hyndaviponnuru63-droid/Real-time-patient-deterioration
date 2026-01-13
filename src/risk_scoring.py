def calculate_risk(row):
    score = 0

    if row["age"] >= 65:
        score += 1
    if row["heart_rate"] > 110:
        score += 2
    if row["oxygen_level"] < 92:
        score += 2
    if row["bp_systolic"] > 140 or row["bp_systolic"] < 90:
        score += 2

    if score >= 5:
        return "High"
    elif score >= 3:
        return "Medium"
    else:
        return "Low"


def explain_risk(row):
    reasons = []

    if row["age"] >= 65:
        reasons.append("Old age")
    if row["heart_rate"] > 110:
        reasons.append("High heart rate")
    if row["oxygen_level"] < 92:
        reasons.append("Low oxygen level")
    if row["bp_systolic"] > 140:
        reasons.append("High blood pressure")
    if row["bp_systolic"] < 90:
        reasons.append("Low blood pressure")

    return ", ".join(reasons) if reasons else "Normal vitals"


def calculate_news(row):
    """
    Simplified NEWS score (educational version)
    """
    score = 0

    # Oxygen saturation
    if row["oxygen_level"] < 92:
        score += 3
    elif row["oxygen_level"] < 95:
        score += 2

    # Heart rate
    if row["heart_rate"] > 130 or row["heart_rate"] < 40:
        score += 3
    elif row["heart_rate"] > 110:
        score += 2

    # Blood pressure
    if row["bp_systolic"] < 90:
        score += 3
    elif row["bp_systolic"] < 100:
        score += 2

    # Age factor
    if row["age"] >= 65:
        score += 1

    return score


def highlight_risk(row):
    if row["risk_level"] == "High":
        return ["background-color: #ffcccc"] * len(row)
    elif row["risk_level"] == "Medium":
        return ["background-color: #fff3cd"] * len(row)
    else:
        return [""] * len(row)
