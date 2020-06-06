# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import math
import datetime

# %% Import data
stock_history = pd.read_csv("./data/dummy-data.csv", delim_whitespace=True)

scaler = MinMaxScaler()
stock_history = scaler.fit_transform(stock_history)
stock_history = pd.DataFrame(stock_history)

# %% Prepare training data
TRAINING_LENGTH = math.ceil(0.7 * len(stock_history))
training_data = stock_history.iloc[:TRAINING_LENGTH, :]

x_train, y_train = [], []
BATCH_SIZE = 3

for i in range(BATCH_SIZE, len(training_data) - 2):
    x_train.append(training_data.iloc[i - BATCH_SIZE : i, 0:1].to_numpy().tolist())
    y_train.append(training_data.iloc[i : i + 1, 1:2].iloc[0, :].to_numpy().tolist())

x_train, y_train = np.array(x_train), np.array(y_train)

# %% Build & Train Model
regressor = Sequential()
regressor.add(
    LSTM(
        units=4,
        activation="relu",
        return_sequences=True,
        input_shape=(x_train.shape[1], x_train.shape[2]),
    )
)
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=10, activation="relu", return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=10, activation="relu"))
regressor.add(Dropout(0.3))
regressor.add(Dense(units=1))

regressor.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"]
)

# %% Train the model
stoppingCallback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2)

regressor.fit(
    x_train, y_train, epochs=40, batch_size=BATCH_SIZE, callbacks=[stoppingCallback]
)

# %% Prepare Test Data & Run Prediction
test_data = stock_history.iloc[TRAINING_LENGTH:, :]

x_test, y_test = [], []

for i in range(BATCH_SIZE, len(test_data)):
    x_test.append(test_data.iloc[i - BATCH_SIZE : i, 0:1].to_numpy().tolist())
    y_test.append(test_data.iloc[i : i + 1, 1:2].iloc[0, :].to_numpy().tolist())

x_test = np.array(x_test)
y_test = np.array(y_test)

#  %% Evaluate & Run Prediction
results = regressor.evaluate(x_test, y_test, batch_size=10)
print(results)

y_predict = regressor.predict(x_test)

x_test = x_test / scaler.scale_[0] + scaler.data_min_[0]
y_predict = y_predict / scaler.scale_[1] + scaler.data_min_[1]

pd.DataFrame(y_predict).plot()


# %%
