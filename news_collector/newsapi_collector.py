# %%
import sys

sys.path.insert(0, "../")
from newsapi.newsapi_client import NewsApiClient
from environs import Env
from datetime import datetime, timedelta
import bisect
from utilities.mongo_connection import get_db_connection

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
        conn = get_db_connection()
        db = Env()("MONGO_DB")

        stocksdb = conn[db]
        modelCollection = stocksdb["news"]

        news_metrics_record = {
            "symbol": symbol,
            "timestamp": timestamp,
            "news_metrics": news_metrics,
        }

        modelCollection.find_one_and_replace(
            {"symbol": symbol, "timestamp": timestamp}, news_metrics_record, upsert=True
        )


if __name__ == "__main__":
    collector = NewsCollector()
    news = collector.collect_n_save_news()
