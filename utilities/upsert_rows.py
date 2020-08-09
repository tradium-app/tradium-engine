# %%
from environs import Env
import logging
import psycopg2
import pandas as pd
from datetime import datetime


def upsert_rows(dataframe, connection):
    cursor = connection.cursor()
    insert_values = dataframe.to_dict(orient="records")
    for row in insert_values:
        query = create_upsert_query(dataframe)
        cursor.execute(query, row)
        connection.commit()
    row_count = len(insert_values)
    logging.info(f"Inserted {row_count} rows.")
    cursor.close()
    del cursor
    connection.close()


def create_upsert_query(dataframe):
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
    env = Env()
    env.read_env()
    DATABASE_URL = env("DATABASE_URL")
    connection = psycopg2.connect(DATABASE_URL)
    print(connection)

    df = pd.DataFrame({"stock": ["MDB"], "datetime": datetime.utcnow()})
    upsert_rows(df, connection)

