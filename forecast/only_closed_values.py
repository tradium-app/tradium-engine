"""
Original file is located at
    https://colab.research.google.com/drive/1GiGU6gShrBbACwNkcKg5aiAj-yxycqkS
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%  Import the libraries
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

df = pd.read_csv("../data/TSLA.csv")
print(df.head())
print(df.shape)

#%% Visualize the closing price history
# plt.figure(figsize=(16, 8))
# plt.title("Close Price History")
df["Close"] = df["closePrice"]
# plt.plot(df["Close"])
# plt.xlabel("Date", fontsize=18)
# plt.ylabel("Close Price USD ($)", fontsize=18)

# Create a new dataframe with only the 'Close' column
data = df.filter(["Close"])
data = data.tail(50000)

#%% Converting the dataframe to a numpy array
dataset = data.values

#%% Get /Compute the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.95)

# Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]

#%% Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60 : i, 0])
    y_train.append(train_data[i, 0])

#%% Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#%% Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer="adam", loss="mae")

#%% Train the model
history = model.fit(x_train, y_train, batch_size=60, epochs=30)

# Test data set
test_data = scaled_data[training_data_len - 60 :, :]

# Create the x_test and y_test data sets
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60 : i, :])

#%% Convert x_test to a numpy array
x_test = np.array(x_test)

#%% Getting the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Undo scaling

#%% Calculate/Get the value of RMSE
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(rmse)

# train = data[:training_data_len]
valid = data[training_data_len:]
valid["Predictions"] = predictions

#%% Visualize the data
# plt.figure(figsize=(16, 8))
# plt.title("Model")
# plt.xlabel("Date", fontsize=18)
# plt.ylabel("Close Price USD ($)", fontsize=18)
# # plt.plot(train['Close'])
# plt.plot(valid[["Close", "Predictions"]])
# # plt.plot(predictions)
# plt.legend(["Train", "Val", "Predictions"], loc="lower right")
# plt.show()

# #%% Show the valid and predicted prices
# print(valid)

# model.save("model.h5")
plt.close()
fig = plt.gcf()

plt.plot(y_test[-500:], color="red", label="Real Price")
plt.plot(predictions[-500:], color="blue", label="Predicted Price")
title = "TSLA Price Prediction:"
plt.title(title)
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc="upper right")
fig.savefig("../charts/tsla-prediction.png", dpi=fig.dpi)
plt.show()
