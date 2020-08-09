#%%  Import the libraries
import math
import numpy as np
import pandas as pd
from environs import Env
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

env = Env()
env.read_env()
DATABASE_URL = env("DATABASE_URL")

engine = create_engine(DATABASE_URL)
df = pd.read_sql(
    "select close_price from stock_data order by datetime desc limit 100000", con=engine
)

df["Close"] = df["close_price"]
data = df.filter(["Close"])

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
BATCH_SIZE = 60

for i in range(BATCH_SIZE, len(train_data) - 6):
    x_train.append(train_data[i - BATCH_SIZE : i, :])
    y_train.append(train_data[i + 6, 0])

#%% Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data into the shape accepted by the LSTM
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#%% Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer="adam", loss="mae")

#%% Train the model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=30)

# Test data set
test_data = scaled_data[training_data_len - BATCH_SIZE :, :]

#%% Create the x_test and y_test data sets
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(BATCH_SIZE, len(test_data)):
    x_test.append(test_data[i - BATCH_SIZE : i, :])


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
plt.close()
fig = plt.gcf()

plt.plot(y_test[-100:], color="red", label="Real Price")
plt.plot(predictions[-100:], color="blue", label="Predicted Price")
title = "TSLA Price Prediction:"
plt.title(title)
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc="upper right")
fig.savefig("../charts/tsla-prediction.png", dpi=fig.dpi)
plt.show()
