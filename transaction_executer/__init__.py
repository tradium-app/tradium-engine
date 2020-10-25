# %%
import sys

sys.path.insert(0, "../")
from environs import Env
import alpaca_trade_api as tradeapi

# from datetime import datetime
# import psycopg2
# import logging

env = Env()
env.read_env()


class TransactionExecuter:
    def execute_prediction(self):
        api = tradeapi.REST(
            key_id=env("APCA_API_KEY_ID"),
            secret_key=env("APCA_API_SECRET_KEY"),
            base_url=env("APCA_API_BASE_URL"),
        )
        order = api.submit_order(
            "NFLX", 1, "buy", "market", "day", None, None, None, False
        )
        print(order)


if __name__ == "__main__":
    executer = TransactionExecuter()
    executer.execute_prediction()
