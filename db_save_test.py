# %%
import sys

sys.path.insert(0, "../")
from environs import Env
from utilities.mongo_connection import get_db_connection
import pandas as pd

# %%
env = Env()
env.read_env()

db = Env()("MONGO_DB")

conn = get_db_connection()
stocksdb = conn[db]
modelCollection = stocksdb["models"]

dataset_train = pd.read_csv("./data/trainset.csv")

# %%

modelCollection.insert_one({"model": dataset_train.to_json(orient="records")})
