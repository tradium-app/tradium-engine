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
    "select close_price from (select datetime, close_price from stock_data order by datetime desc limit 10000) as temp order by datetime asc",
    con=engine,
)

df["Close"] = df["close_price"]
data = df.filter(["Close"])

dataset = data.values

#%% Get /Compute the number of rows to train the model on
training_data_len = len(dataset)

# Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]

#%% Split the data into x_train and y_train data sets
x_train = []
y_train = []
BATCH_SIZE = 60
NEXT_PREDICTION_STEP = 6

for i in range(BATCH_SIZE, len(train_data) - NEXT_PREDICTION_STEP):
    x_train.append(train_data[i - BATCH_SIZE : i, :])
    y_train.append(train_data[i + NEXT_PREDICTION_STEP, 0])

#%% Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#%% Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer="adam", loss="mae")

# %% Train the model
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=30)

# %% Run Predictions
x_test = train_data[-BATCH_SIZE:]
x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Undo scaling

# %% Save Predictions
print(predictions)
