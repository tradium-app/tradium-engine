# %%
import sys

sys.path.insert(0, "../")
from environs import Env
import alpaca_trade_api as tradeapi

from datetime import datetime
import psycopg2
from decimal import Decimal

env = Env()
env.read_env()


class TransactionExecuter:
    def execute_prediction(self):
        current_price, predicted_price = self.fetch_prediction()

        if predicted_price > Decimal(1.05) * current_price:
            api = tradeapi.REST(
                key_id=env("APCA_API_KEY_ID"),
                secret_key=env("APCA_API_SECRET_KEY"),
                base_url=env("APCA_API_BASE_URL"),
            )
            order = api.submit_order(
                "TSLA", 1, "buy", "market", "day", None, None, None, False
            )
            print(order)
        else:
            print(
                f"Latest Price: {current_price} \nPredicted Price: {predicted_price}.\nTrade not executed at {datetime.now()}"
            )

    def fetch_prediction(self):
        DATABASE_URL = env("DATABASE_URL")
        connection = psycopg2.connect(DATABASE_URL)

        cursor = connection.cursor()

        sql_select_query = """select datetime,close_price,predicted_close_price from stock_data order by datetime desc limit 5"""
        cursor.execute(sql_select_query, (5,))
        record = cursor.fetchall()

        predicted_price = record[0][2]
        current_price = next(filter(lambda x: not x[1] is None, record), None)[1]

        return (current_price, predicted_price)


if __name__ == "__main__":
    executer = TransactionExecuter()
    executer.execute_prediction()
