import sys

sys.path.insert(0, "../")
import gc
import time
from environs import Env
from utilities.mongo_connection import get_db_connection
from pytz import utc
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from database.seeding import Seeding
from predictors.predictor import Predictor
from news_collector.newsapi_collector import NewsCollector
from tweets_collector.tweets_collector import TweetsCollector

env = Env()
env.read_env()
DATABASE_URL = env("DATABASE_URL")

Seeding().initialize()

jobstores = {
    "default": SQLAlchemyJobStore(url=DATABASE_URL, tablename="apscheduler_jobs"),
}

executors = {"default": ThreadPoolExecutor(20), "processpool": ProcessPoolExecutor(5)}

job_defaults = {"coalesce": False, "max_instances": 3}
scheduler = BackgroundScheduler(
    jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone=utc
)

scheduler.add_job(
    Predictor().predict_and_save,
    "interval",
    minutes=10,
    id="prediction_job",
    replace_existing=True,
)

scheduler.add_job(
    NewsCollector().collect_n_save_news,
    "interval",
    minutes=30,
    id="news_job",
    replace_existing=True,
)

scheduler.add_job(
    TweetsCollector().collect_n_save_tweets,
    "interval",
    minutes=30,
    id="tweet_job",
    replace_existing=True,
)

scheduler.start()

while True:
    time.sleep(10)
    gc.collect()
