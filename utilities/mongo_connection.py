# %%
import pymongo
import dns
from environs import Env


def get_db_connection():
    env = Env()
    db_url = env("MONGODB_URI")
    dbClient = pymongo.MongoClient(db_url)

    return dbClient
