# https://www.kaggle.com/ptheru/google-stock-price-prediction-rnn
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# %%
dataset_train = pd.read_csv("./data/trainset.csv")

trainset = dataset_train.iloc[:, 1:2].values

# %%
sc = MinMaxScaler(feature_range=(0, 1))
training_scaled = sc.fit_transform(trainset)

x_train, y_train = [], []
BATCH_SIZE = 60

for i in range(BATCH_SIZE, len(training_scaled)):
    x_train.append(training_scaled[i - BATCH_SIZE : i, 0:1])
    y_train.append(training_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)


# %% Build the model

regressor = Sequential()

regressor.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(x_train.shape[1], x_train.shape[2]),
    )
)
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer="adam", loss="mean_squared_error")

# %% Train the Model
TRAIN_BATCH_SIZE = 32
regressor.fit(x_train, y_train, epochs=10, batch_size=TRAIN_BATCH_SIZE)

# %%
dataset_test = pd.read_csv("./data/testset.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# %% Prepare Test
x_test = []
for i in range(BATCH_SIZE, len(inputs)):
    x_test.append(inputs[i - BATCH_SIZE : i, 0:1])

x_test = np.array(x_test)

# %%
predicted_price = regressor.predict(x_test)
predicted_price = sc.inverse_transform(predicted_price)

# %%
plt.plot(real_stock_price, color="red", label="Real Price")
plt.plot(predicted_price, color="blue", label="Predicted Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
