import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_lstm(df):
    scaler = MinMaxScaler()

    X = df.drop(['death_inhosp'], axis=1)
    y = df['death_inhosp'].values

    X_scaled = scaler.fit_transform(X.values)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = Sequential()
    model.add(LSTM(32, input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(X_scaled, y, epochs=5, batch_size=16, verbose=0)

    return model, scaler


def predict_lstm(model, scaler, df_row):
    """
    df_row: pandas DataFrame with ONLY numeric columns (1 row)
    """
    X = df_row.values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.reshape((1, 1, X_scaled.shape[1]))

    prediction = model.predict(X_scaled, verbose=0)
    return float(prediction[0][0])
