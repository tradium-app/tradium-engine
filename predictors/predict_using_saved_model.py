# %% Import libraries
from os import write
from keras.models import load_model
from environs import Env
import psycopg2

# %% Load the model from postgres db
env = Env()
env.read_env()
DATABASE_URL = env("DATABASE_URL")
connection = psycopg2.connect(DATABASE_URL)
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
