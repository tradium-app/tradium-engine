# %%
import sys

sys.path.insert(0, "../")
import psycopg2
from newsapi.newsapi_client import NewsApiClient
from environs import Env
from datetime import datetime, timedelta
import bisect

env = Env()
env.read_env()


class NewsCollector:
    def collect_n_save_news(self):
        company = "Tesla"
        symbol = "TSLA"

        api_key = env("NEWSAPI_KEY")

        newsapi = NewsApiClient(api_key=api_key)

        start_time = self.rounded_to_the_last_30th_minute_epoch()
        end_time = start_time + timedelta(minutes=30)

        news = newsapi.get_everything(
            q=company,
            language="en",
            from_param=start_time,
            to=end_time,
            sort_by="popularity",
            page_size=100,
            page=1,
        )

        self.save_news_metrics(symbol, end_time, {"news_count": len(news["articles"])})

    def rounded_to_the_last_30th_minute_epoch(self):
        now = datetime.now()
        rounded = now - (now - datetime.min) % timedelta(minutes=30)
        return rounded

    def save_news_metrics(self, symbol, timestamp, news_metrics):
        DATABASE_URL = env("DATABASE_URL")
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()

        postgres_insert_query = """INSERT INTO Stock_Data (Stock, DateTime, News_Count)
        VALUES (%s,%s,%s)"""
        record_to_insert = ("DAL", timestamp, news_metrics["news_count"])
        cursor.execute(postgres_insert_query, record_to_insert)
        connection.commit()


if __name__ == "__main__":
    collector = NewsCollector()
    collector.collect_n_save_news()
