# %%
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from environs import Env


class TwitterClient(object):
    def __init__(self):
        env = Env()
        consumer_key = env("TWITTER_CONSUMER_KEY")
        consumer_secret = env("TWITTER_CONSUMER_SECRET")
        access_token = env("TWITTER_ACCESS_TOKEN")
        access_token_secret = env("TWITTER_ACCESS_TOKEN_SECRET")

        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)

            self.api = tweepy.API(self.auth)
        except Exception as exception:
            print(f"Auth Error: {exception}")

    def clean_tweet(self, tweet):
        return " ".join(
            re.sub(
                "(@[A-Za-z0-9]+) | ([ ^ 0-9A-Za-z \t]) | (\w+: \/\/\S+)", " ", tweet
            ).split()
        )

    def get_tweet_sentiment(self, tweet):
        clean_tweet = self.clean_tweet(tweet)
        analysis = TextBlob(clean_tweet)
        if analysis.sentiment.polarity > 0:
            return "positive"
        elif analysis.sentiment.polarity == 0:
            return "neutral"
        else:
            return "negative"

    def get_tweets(self, query, count=10):
        tweets = []
        # try:
        fetched_tweets = self.api.search(q=query, count=count)
        for tweet in fetched_tweets:
            parsed_tweet = {}
            parsed_tweet["text"] = tweet.text
            parsed_tweet["sentiment"] = self.get_tweet_sentiment(tweet.text)
            parsed_tweet["polarity"] = ""

            if tweet.retweet_count > 0:
                if parsed_tweet not in tweets:
                    tweets.append(parsed_tweet)
            else:
                tweets.append(parsed_tweet)

        return tweets
        # except Exception as exception:
        #     print(f'Error: {exception}')


# %%
# client = TwitterClient()
# client.get_tweets('donald')

# good_sentiment = client.get_tweet_sentiment('what a beautiful day')
# print(good_sentiment)

# bad_sentiment = client.get_tweet_sentiment('what a bad day')
# print(bad_sentiment)
