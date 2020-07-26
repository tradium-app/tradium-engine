# %% Import libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import math
import matplotlib.pyplot as plt

# %% Import data

cols_list = [
    "startEpochTime",
    "openPrice",
    "highPrice",
    "lowPrice",
    "volume",
    "closePrice",
]

stock_history = pd.read_csv("../data/TSLA.csv")
stock_history = stock_history[cols_list]

scaler = MinMaxScaler()
stock_history = scaler.fit_transform(stock_history)
stock_history = pd.DataFrame(stock_history)
stock_history = stock_history.head(1000)

# %% Prepare training data
TRAINING_LENGTH = math.ceil(0.85 * len(stock_history))
training_data = stock_history.iloc[:TRAINING_LENGTH, :]

x_train, y_train = [], []
BATCH_SIZE = 30
NO_OF_INPUT_COLS = 5

for i in range(BATCH_SIZE, len(training_data) - 2):
    x_train.append(
        training_data.iloc[i - BATCH_SIZE : i, 0:NO_OF_INPUT_COLS].to_numpy().tolist()
    )
    y_train.append(
        training_data.iloc[i : i + 1, NO_OF_INPUT_COLS : NO_OF_INPUT_COLS + 1]
        .iloc[0, :]
        .to_numpy()
        .tolist()
    )

x_train, y_train = np.array(x_train), np.array(y_train)

# %% Build Model
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
stoppingCallback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)

regressor.fit(
    x_train, y_train, epochs=100, batch_size=BATCH_SIZE, callbacks=[stoppingCallback]
)

# %% Prepare Test Data & Run Prediction
test_data = stock_history.iloc[TRAINING_LENGTH:, :]

x_test, y_test = [], []

for i in range(BATCH_SIZE, len(test_data)):
    x_test.append(
        test_data.iloc[i - BATCH_SIZE : i, 0:NO_OF_INPUT_COLS].to_numpy().tolist()
    )
    y_test.append(
        test_data.iloc[i : i + 1, NO_OF_INPUT_COLS : NO_OF_INPUT_COLS + 1]
        .iloc[0, :]
        .to_numpy()
        .tolist()
    )

x_test = np.array(x_test)
y_test = np.array(y_test)

#  %% Evaluate & Run Prediction

test_score, test_acc = regressor.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print(test_score, test_acc)

y_predict = regressor.predict(x_test)

# x_test = x_test / scaler.scale_[0] + scaler.data_min_[0]
y_predict = (
    y_predict / scaler.scale_[NO_OF_INPUT_COLS] + scaler.data_min_[NO_OF_INPUT_COLS]
)
y_test = y_test / scaler.scale_[NO_OF_INPUT_COLS] + scaler.data_min_[NO_OF_INPUT_COLS]


# %%
fig = plt.gcf()
plt.plot(y_test, color="red", label="Real Price")
plt.plot(y_predict, color="blue", label="Predicted Price")
title = "TSLA Price Prediction: test_score: {test_score:.3f}, test_acc: {test_acc:.3f}".format(
    test_score=test_score, test_acc=test_acc
)

plt.title(title)
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
fig.savefig("../charts/tsla-prediction.png", dpi=fig.dpi)
plt.show()
