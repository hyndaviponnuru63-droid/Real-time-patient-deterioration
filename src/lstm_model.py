import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_lstm(df, feature_cols):
    X = df[feature_cols].values
    y = df["death_inhosp"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = Sequential([
        LSTM(32, input_shape=(1, X_scaled.shape[2])),
        Dense(1, activation="sigmoid")
    ])

    model.compile("adam", "binary_crossentropy")
    model.fit(X_scaled, y, epochs=3, batch_size=32, verbose=0)

    return model, scaler

def predict_lstm(model, scaler, feature_cols, df_row):
    X = df_row[feature_cols].values
    X = scaler.transform(X)
    X = X.reshape((1, 1, X.shape[1]))
    return float(model.predict(X, verbose=0)[0][0])
