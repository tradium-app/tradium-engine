# %%
import sys

sys.path.insert(0, "../")
from environs import Env
import alpaca_trade_api as tradeapi
from datetime import datetime
import psycopg2
import pandas as pd
import logging

env = Env()
env.read_env()


class StocksRefresher:
    def refresh(self):
        bars = self.pull_latest()
        self.save(bars)

    def pull_latest(self):
        api = tradeapi.REST()
        bars = api.get_barset(
            ["TSLA"], "5Min", limit=4, until=datetime.utcnow().isoformat()
        )
        return bars["TSLA"].df

    def save(self, dataframe):
        dataframe.reset_index(level=0, inplace=True)

        dataframe = dataframe.rename(
            columns={
                "time": "datetime",
                "open": "open_price",
                "high": "high_price",
                "low": "low_price",
                "close": "close_price",
                "volume": "volume",
            }
        )
        dataframe["stock"] = "TSLA"

        DATABASE_URL = env("DATABASE_URL")
        connection = psycopg2.connect(DATABASE_URL)

        self.upsert_rows(dataframe, connection)
        return

    def upsert_rows(self, dataframe, connection):
        cursor = connection.cursor()
        insert_values = dataframe.to_dict(orient="records")
        for row in insert_values:
            query = self.create_upsert_query(dataframe)
            cursor.execute(query, row)
            connection.commit()
        row_count = len(insert_values)
        logging.info(f"Inserted {row_count} rows.")
        cursor.close()
        del cursor
        connection.close()

    def create_upsert_query(self, dataframe):
        columns = ", ".join([f"{col}" for col in dataframe.columns])
        constraint = ", ".join([f"{col}" for col in ["stock", "datetime"]])
        placeholder = ", ".join([f"%({col})s" for col in dataframe.columns])
        updates = ", ".join([f"{col} = EXCLUDED.{col}" for col in dataframe.columns])

        query = f"""INSERT INTO stock_data_2 ({columns})
                    VALUES ({placeholder})
                    ON CONFLICT ({constraint})
                    DO UPDATE SET {updates};"""
        query.split()
        query = " ".join(query.split())
        return query


if __name__ == "__main__":
    refresher = StocksRefresher()
    refresher.refresh()
