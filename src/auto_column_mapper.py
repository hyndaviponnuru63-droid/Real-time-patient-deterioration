def auto_map_columns(df):
    aliases = {
        "caseid": ["caseid", "patient_id", "id"],
        "age": ["age", "patient_age"],
        "sex": ["sex", "gender"],
        "asa": ["asa", "asa_score"],
        "icu_days": ["icu_days", "icu_stay", "icu_days_stay"],
        "emop": ["emop", "emergency", "emergency_op"],
        "preop_hb": ["preop_hb", "hb", "hemoglobin"],
        "preop_na": ["preop_na", "sodium", "na"],
        "preop_gluc": ["preop_gluc", "glucose", "blood_glucose"]
    }

    mapped = {}

    for std_col, possible_names in aliases.items():
        for col in df.columns:
            if col.lower() in possible_names:
                mapped[col] = std_col
                break

    df = df.rename(columns=mapped)

    missing = [c for c in aliases if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[list(aliases.keys())]
