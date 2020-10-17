# %% Import libraries
import sys

sys.path.insert(0, "../")
from keras.models import load_model
from environs import Env
import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from utilities.upsert_rows import upsert_rows

# %% Initialize
env = Env()
env.read_env()
DATABASE_URL = env("DATABASE_URL")
connection = psycopg2.connect(DATABASE_URL)

# %% Load the model from postgres db
cursor = connection.cursor()

cursor.execute(
    "select model from stock_model where stock=%s order by datetime desc limit 1",
    ("TSLA",),
)
model_file_binary = cursor.fetchone()[0]

model_file_name = "lstm_model.h5"
model_file_io = open(model_file_name, "wb")
model_file_io.write(model_file_binary.tobytes())

# %% Load the model to keras
model = load_model(model_file_name)

# %% Fetch latest data for prediction
engine = create_engine(DATABASE_URL)
BATCH_SIZE = 60
NEXT_PREDICTION_STEP = 30

df = pd.read_sql(
    "select datetime, close_price from (select datetime, close_price from stock_data where close_price is not null order by datetime desc limit %s ) as temp order by datetime asc"
    % BATCH_SIZE,
    con=engine,
)

# %% Run prediction
x_test = df.iloc[:, 1:2].to_numpy()

scaler = MinMaxScaler(feature_range=(0, 1))
x_test_scaled = scaler.fit_transform(x_test)
x_test_scaled = np.reshape(
    x_test_scaled, (1, x_test_scaled.shape[0], x_test_scaled.shape[1])
)

x_predictions = model.predict(x_test_scaled)
x_predictions = scaler.inverse_transform(x_predictions)


# %% Save Predictions
predicted_date = df.tail(1)["datetime"].values[0] + np.timedelta64(
    NEXT_PREDICTION_STEP, "m"
)

print(f"Predicted price for {predicted_date}: {x_predictions[0][0]}")

predict_df = pd.DataFrame(
    {
        "stock": ["TSLA"],
        "datetime": [predicted_date],
        "predicted_close_price": [x_predictions[0][0]],
    }
)

upsert_rows("stock_data", predict_df, connection)
