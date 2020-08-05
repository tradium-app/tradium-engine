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
# df = df.tail(5000)

dataset = df.values

#%% Prepare training data
training_data_len = math.ceil(len(dataset) * 0.95)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len]

#%% Prepare x_train and y_train
BATCH_SIZE = 60
x_train = []
y_train = []


for i in range(BATCH_SIZE, len(train_data)):
    x_train.append(train_data[i - BATCH_SIZE : i])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#%% Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mae")

#%% Train the model
stoppingCallback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=30,
    callbacks=[stoppingCallback],
    shuffle=False,
)

#%% Prepare Test data
test_data = scaled_data[training_data_len - BATCH_SIZE :, :]

x_test = []
# y_test = []

for i in range(BATCH_SIZE, len(test_data)):
    x_test.append(test_data[i - BATCH_SIZE : i])
    # y_test.append(test_data[i, 0])

x_test = np.array(x_test)
# x_test, y_test = np.array(x_test), np.array(y_test)

y_test = dataset[training_data_len:, :]

#%% Evaluate Model

model_evaluation_result = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print(model_evaluation_result)

# %% Run Prediction
y_predict = model.predict(x_test)
y_predict = scaler.inverse_transform(y_predict)

#%% Calculate root mean square error
rmse = np.sqrt(np.mean(((y_predict - y_test) ** 2)))
print(rmse)

#%% Plot results
plt.close()
fig = plt.gcf()

plt.plot(y_test, color="red", label="Real Price")
plt.plot(y_predict, color="blue", label="Predicted Price")
title = "TSLA Price Prediction: test_score: {result:.3f}".format(
    result=model_evaluation_result
)
plt.title(title)
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc="upper right")
fig.savefig("../charts/tsla-prediction.png", dpi=fig.dpi)
plt.show()
