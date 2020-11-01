#%%  Import the libraries
from datetime import date
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
import psycopg2
import datetime

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
BATCH_SIZE = 60
x_train = []
y_train = []
NEXT_PREDICTION_STEP = 6

for i in range(BATCH_SIZE, len(train_data) - NEXT_PREDICTION_STEP):
    x_train.append(train_data[i - BATCH_SIZE : i, :])
    y_train.append(train_data[i + NEXT_PREDICTION_STEP, 0])

#%% Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
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
# stoppingCallback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=30,
    # callbacks=[stoppingCallback],
    shuffle=False,
)

# %% Save the model to file system
model_file_name = "lstm_model.h5"
model.save(model_file_name, overwrite=True)

# %% Save the model to postgres db
env = Env()
env.read_env()
DATABASE_URL = env("DATABASE_URL")
connection = psycopg2.connect(DATABASE_URL)

model_file_io = open(model_file_name, "rb")
model_file = model_file_io.read()
model_file_binary = psycopg2.Binary(model_file)

cursor = connection.cursor()

cursor.execute(
    "insert into stock_model(stock, datetime, model) values (%s, %s, %s)",
    ("TSLA", datetime.datetime.utcnow(), model_file_binary),
)
connection.commit()
