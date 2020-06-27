# %%
import psycopg2
from environs import Env

# %%

env = Env()
env.read_env()


class Seeding:
    def initialize(self):
        try:
            DATABASE_URL = env("DATABASE_URL")
            connection = psycopg2.connect(DATABASE_URL)

            cursor = connection.cursor()

            create_table_query = """CREATE TABLE IF NOT EXISTS Stock_Data
                (ID           INT PRIMARY KEY     NOT NULL,
                Stock         TEXT    NOT NULL,
                DateTime      timestamptz NOT NULL,
                Tweet_Count             INT,
                Positive_Tweet_Count    INT,
                Negative_Tweet_Count    INT); """
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
