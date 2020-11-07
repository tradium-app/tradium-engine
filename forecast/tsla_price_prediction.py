# %% Import libraries
import math
import numpy as np
import pandas as pd
import random as rn
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

rn.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

plt.style.use("fivethirtyeight")

#%% Import dataframe

df = pd.read_csv("../data/TSLA.csv")
df = df.filter(["closePrice"])
df = df.tail(10000)

dataset = df.values
features_length = dataset.shape[1]

#%% Prepare training data
training_data_len = math.ceil(len(dataset) * 0.80)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]

#%% Prepare x_train and y_train
BATCH_SIZE = 120
x_train = []
y_train = []
NEXT_PREDICTION_STEP = 6

for i in range(BATCH_SIZE, len(train_data) - NEXT_PREDICTION_STEP):
    x_train.append(train_data[i - BATCH_SIZE : i, :])
    y_train.append(train_data[i + NEXT_PREDICTION_STEP, -1:])

x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#%% Build the LSTM network model
model = Sequential()
model.add(
    LSTM(
        units=50, return_sequences=True, input_shape=(x_train.shape[1], features_length)
    )
)
model.add(
    LSTM(
        units=50, return_sequences=True, input_shape=(x_train.shape[1], features_length)
    )
)
model.add(
    LSTM(
        units=50, return_sequences=True, input_shape=(x_train.shape[1], features_length)
    )
)
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mae")

#%% Train the model
stoppingCallback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=30,
    callbacks=[stoppingCallback],
    shuffle=False,
)

#%% Prepare Test data
test_data = scaled_data[training_data_len - BATCH_SIZE :, :]

x_test = []
y_test = dataset[training_data_len + NEXT_PREDICTION_STEP :, features_length - 1]
y_test_previous = dataset[training_data_len:-NEXT_PREDICTION_STEP, features_length - 1]

for i in range(BATCH_SIZE, len(test_data) - NEXT_PREDICTION_STEP):
    x_test.append(test_data[i - BATCH_SIZE : i, :])

x_test = np.array(x_test)

# %% Run Prediction
y_predict = model.predict(x_test)

dummy = np.zeros(shape=(len(y_predict), features_length))
dummy[:, features_length - 1] = np.reshape(y_predict, (y_predict.shape[0],))

y_predict = scaler.inverse_transform(dummy)[:, [features_length - 1]]

#%% Calculate root mean square error
error = sum(abs(y_predict - y_test))[0]

profit_or_loss = [0]
correct_prediction = 0
total_prediction = 0
for i in range(0, len(y_test)):
    if y_predict[i] > y_test_previous[i]:
        profit_or_loss += y_test[i] - y_test_previous[i]
    if (y_test[i] > y_test_previous[i] and y_predict[i] > y_test_previous[i]) or (
        y_test[i] < y_test_previous[i] and y_predict[i] > y_test_previous[i]
    ):
        correct_prediction += 1

profit_or_loss = profit_or_loss[0]
prediction_accuracy = (correct_prediction * 100) / len(y_test)

#%% Plot results
plt.close()
fig = plt.gcf()

plt.plot(y_test[-2000:], color="red", label="Real Price")
# plt.scatter()
plt.plot(y_predict[-2000:], color="blue", label="Predicted Price")
title = "TSLA Price Prediction: error: {error:.2f}, profit_or_loss: {profit_or_loss:.2f}, prediction_accuracy: {prediction_accuracy: .2f}%".format(
    error=error, profit_or_loss=profit_or_loss, prediction_accuracy=prediction_accuracy
)
plt.title(title)
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc="upper right")
# fig.savefig("../charts/tsla-prediction.png", dpi=fig.dpi)
plt.show()
