import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_lstm(df):
    """
    df: preprocessed dataframe for LSTM
    """
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.drop(['death_inhosp'], axis=1))
    y = df['death_inhosp'].values

    # Reshape for LSTM [samples, timesteps, features]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    return model, scaler

def predict_lstm(model, scaler, df_row):
    X = scaler.transform(df_row)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    return model.predict(X)[0][0]
