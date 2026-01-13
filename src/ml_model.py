from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model(df):
    features = [
        "age", "asa", "icu_days", "emop",
        "preop_hb", "preop_na", "preop_gluc"
    ]

    df["target"] = (df["icu_days"] > 0).astype(int)

    X = df[features]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, scaler, accuracy, X_train


def predict_risk_probability(model, scaler, df):
    features = [
        "age", "asa", "icu_days", "emop",
        "preop_hb", "preop_na", "preop_gluc"
    ]

    X_scaled = scaler.transform(df[features])
    df["risk_probability"] = model.predict_proba(X_scaled)[:, 1]

    return df
