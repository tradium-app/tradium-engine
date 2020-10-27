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
DATABASE_URL = env("DATABASE_URL")


class TransactionExecuter:
    def execute_prediction(self):
        connection = psycopg2.connect(DATABASE_URL)
        current_price, predicted_price = self.fetch_prediction(connection)
        current_position = self.fetch_positions(connection, "TSLA")

        api = tradeapi.REST(
            key_id=env("APCA_API_KEY_ID"),
            secret_key=env("APCA_API_SECRET_KEY"),
            base_url=env("APCA_API_BASE_URL"),
        )

        if current_position == 0 and predicted_price > Decimal(1.01) * current_price:
            order = api.submit_order(
                "TSLA", 1, "buy", "market", "day", None, None, None, False
            )
            print(order)
        elif current_position > 0 and predicted_price < Decimal(0.99) * current_price:
            order = api.submit_order(
                "TSLA", 1, "sell", "market", "day", None, None, None, False
            )
            print(order)
        else:
            print(
                f"Latest Price: {current_price} \nPredicted Price: {predicted_price}.\nTrade not executed at {datetime.now()}"
            )

    def fetch_prediction(self, connection):
        cursor = connection.cursor()

        sql_select_query = """select datetime,close_price,predicted_close_price from stock_data order by datetime desc limit 5"""
        cursor.execute(sql_select_query, (5,))
        record = cursor.fetchall()

        predicted_price = record[0][2]
        current_price = next(filter(lambda x: not x[1] is None, record), None)[1]

        return (current_price, predicted_price)

    def fetch_positions(self, connection, stock):
        cursor = connection.cursor()

        sql_select_query = f"""select stock,positions from stock_positions where LOWER(stock)=LOWER('{stock}');"""
        print(sql_select_query)

        cursor.execute(sql_select_query)

        current_position = cursor.fetchone()[1]
        return current_position

    def update_positions(self, connection, stock, new_positions):
        cursor = connection.cursor()
        upsert_query = f"""INSERT INTO stock_positions (stock,positions)
                            VALUES('{stock}', '{new_positions}')
                            ON CONFLICT(stock)
                            DO UPDATE SET positions='{new_positions}';"""

        upsert_query.split()
        upsert_query = " ".join(upsert_query.split())
        print(upsert_query)
        cursor.execute(upsert_query)
        connection.commit()

        pass


if __name__ == "__main__":
    executer = TransactionExecuter()
    connection = psycopg2.connect(DATABASE_URL)
    positions = executer.fetch_positions(connection, "TSLA")
    print(positions)
