# %%
import pandas as pd
from environs import Env
from sqlalchemy import create_engine

df = pd.read_csv("../forecast/TSLA.csv")

df.columns = [c.lower() for c in df.columns]  # postgres doesn't like capitals or spaces
df["stock"] = "TSLA"
df["datetime"] = pd.to_datetime(
    df["start_epoch_time"], utc=True, unit="s", origin="unix"
)

env = Env()
env.read_env()
DATABASE_URL = env("DATABASE_URL")

engine = create_engine(DATABASE_URL)

df.to_sql("stock_data", engine, if_exists="append", index=False, chunksize=100)
