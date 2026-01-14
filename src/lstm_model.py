import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_lstm(df):
    scaler = MinMaxScaler()
    feature_cols = df.drop(['death_inhosp'], axis=1).columns.tolist()
    X = df[feature_cols].values
    y = df['death_inhosp'].values

    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = Sequential()
    model.add(LSTM(32, input_shape=(1, X_scaled.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_scaled, y, epochs=5, batch_size=16, verbose=0)
    return model, scaler, feature_cols

def predict_lstm(model, scaler, feature_cols, df_row):
    row = df_row.copy()

    # Ensure all features exist
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    # Fill missing values
    row = row[feature_cols].fillna(0)

    X = row.values
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((1, 1, X_scaled.shape[1]))

    risk = model.predict(X_scaled, verbose=0)[0][0]

    # Safety clamp
    if np.isnan(risk):
        return 0.0

    return float(np.clip(risk, 0, 1))

