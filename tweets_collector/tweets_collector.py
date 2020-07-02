# %%
import os
import sys

sys.path.insert(0, "../")
import psycopg2
from environs import Env
from datetime import datetime, timedelta
from tweets_collector.TwitterClient import TwitterClient


env = Env()
env.read_env()


class TweetsCollector:
    def collect_n_save_tweets(self):
        company = "Delta Air Lines"
        symbol = "DAL"
        searchTerm = "$DAL"

        api = TwitterClient()
        tweets = api.get_tweets(query=searchTerm, count=2000)

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
            company,
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

    def save_tweets_metrics(self, symbol, company, timestamp, tweets_metrics):
        DATABASE_URL = env("DATABASE_URL")
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()

        postgres_insert_query = """INSERT INTO Stock_Data (Stock, Company, DateTime, Tweets_Count, Positive_Tweets_Count, Negative_Tweets_Count, news_count)
        VALUES (%s,%s,%s,%s,%s,%s, %s)"""
        record_to_insert = (
            symbol,
            company,
            timestamp,
            tweets_metrics["positive_tweets_count"]
            + tweets_metrics["negative_tweets_count"],
            tweets_metrics["positive_tweets_count"],
            tweets_metrics["negative_tweets_count"],
            0,
        )

        cursor.execute(postgres_insert_query, record_to_insert)
        connection.commit()


if __name__ == "__main__":
    collector = TweetsCollector()
    collector.collect_n_save_tweets()
