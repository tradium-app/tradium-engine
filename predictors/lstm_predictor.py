#%%  Import the libraries
import sys

sys.path.insert(0, "../")
import numpy as np
import pandas as pd
import random as rn
from environs import Env
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from utilities.upsert_rows import upsert_rows
import psycopg2

rn.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
plt.style.use("fivethirtyeight")

env = Env()
env.read_env()
DATABASE_URL = env("DATABASE_URL")

engine = create_engine(DATABASE_URL)
df = pd.read_sql(
    "select datetime, close_price from (select datetime, close_price from stock_data where close_price is not null order by datetime desc limit 50000) as temp order by datetime asc",
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
NEXT_PREDICTION_STEP = 2

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
predicted_date = df.tail(1)["datetime"].values[0] + np.timedelta64(
    NEXT_PREDICTION_STEP * 5, "m"
)

print(f"Predicted price for {predicted_date}: {predictions[0][0]}")

env = Env()
env.read_env()
DATABASE_URL = env("DATABASE_URL")
connection = psycopg2.connect(DATABASE_URL)

predict_df = pd.DataFrame(
    {
        "stock": ["TSLA"],
        "datetime": [predicted_date],
        "predicted_close_price": [predictions[0][0]],
    }
)

upsert_rows("stock_data", predict_df, connection)
