def calculate_risk(row):
    score = 0

    if row["age"] >= 65:
        score += 2
    if row["asa"] >= 3:
        score += 3
    if row["emop"] == 1:
        score += 2
    if row["icu_days"] >= 1:
        score += 2
    if row["preop_hb"] < 10:
        score += 1
    if row["preop_na"] < 130 or row["preop_na"] > 150:
        score += 1
    if row["preop_gluc"] > 180:
        score += 1

    if score >= 6:
        return "High"
    elif score >= 3:
        return "Medium"
    else:
        return "Low"


def explain_risk(row):
    reasons = []

    if row["age"] >= 65:
        reasons.append("Elderly patient")
    if row["asa"] >= 3:
        reasons.append("High ASA score")
    if row["emop"] == 1:
        reasons.append("Emergency surgery")
    if row["icu_days"] >= 1:
        reasons.append("ICU admission")
    if row["preop_hb"] < 10:
        reasons.append("Low hemoglobin")
    if row["preop_gluc"] > 180:
        reasons.append("High glucose")

    return ", ".join(reasons) if reasons else "No major risk factors"


def highlight_risk(row):
    if row["risk_level"] == "High":
        return ["background-color: #ffcccc"] * len(row)
    elif row["risk_level"] == "Medium":
        return ["background-color: #fff3cd"] * len(row)
    return [""] * len(row)
