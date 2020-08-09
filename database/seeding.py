# %%
import psycopg2
from environs import Env

# %%

env = Env()
env.read_env()


class Seeding:
    def initialize(self):
        DATABASE_URL = env("DATABASE_URL")
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()

        try:
            create_table_query = """CREATE TABLE IF NOT EXISTS Stock_Data
                (id                      SERIAL PRIMARY KEY NOT NULL,
                stock                    TEXT    NOT NULL,
                company                  TEXT,
                datetime                 timestamptz NOT NULL,
                start_epoch_time         INT,
                open_price               NUMERIC,
                high_price               NUMERIC,
                low_price                NUMERIC,
                close_price              NUMERIC,
                volume                   INT,
                news_count               INT,
                tweets_count             INT,
                positive_tweets_count    INT,
                negative_tweets_count    INT,
                predicted_close_price    NUMERIC,
                error                    NUMERIC); """
            cursor.execute(create_table_query)
            connection.commit()

        except (Exception, psycopg2.DatabaseError) as error:
            print("Error while creating PostgreSQL table", error)
        finally:
            # closing database connection.
            if connection:
                cursor.close()
                connection.close()
                print("PostgreSQL connection is closed")


if __name__ == "__main__":
    seeding = Seeding()
    seeding.initialize()
