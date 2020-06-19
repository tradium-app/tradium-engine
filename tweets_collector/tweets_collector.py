# %%
import os
import sys

sys.path.insert(0, "../")
from environs import Env
from datetime import datetime, timedelta
from utilities.mongo_connection import get_db_connection
from tweets_collector.TwitterClient import TwitterClient


env = Env()
env.read_env()


class TweetsCollector:
    def collect_n_save_tweets(self):
        company = "Delta Air Lines"
        symbol = "DAL"

        api = TwitterClient()
        tweets = api.get_tweets(query=company, count=2000)

        positive_tweets = [
            tweet for tweet in tweets if tweet["sentiment"] == "positive"
        ]

        negative_tweets = [
            tweet for tweet in tweets if tweet["sentiment"] == "negative"
        ]

        start_time = self.rounded_to_the_last_30th_minute_epoch()
        end_time = start_time + timedelta(minutes=30)

        self.save_tweets_metrics(
            symbol,
            end_time,
            {
                "positive_tweets_count": len(positive_tweets),
                "negative_tweets_count": len(negative_tweets),
            },
        )

    def rounded_to_the_last_30th_minute_epoch(self):
        now = datetime.now()
        rounded = now - (now - datetime.min) % timedelta(minutes=30)
        return rounded

    def save_tweets_metrics(self, symbol, timestamp, tweets_metrics):
        conn = get_db_connection()
        db = Env()("MONGO_DB")

        stocksdb = conn[db]
        modelCollection = stocksdb["tweets"]

        tweets_metrics_record = {
            "symbol": symbol,
            "timestamp": timestamp,
            "tweets_metrics": tweets_metrics,
        }

        modelCollection.find_one_and_replace(
            {"symbol": symbol, "timestamp": timestamp},
            tweets_metrics_record,
            upsert=True,
        )


if __name__ == "__main__":
    collector = TweetsCollector()
    collector.collect_n_save_tweets()
