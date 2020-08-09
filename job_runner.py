# %%
from datetime import datetime
import sys

sys.path.insert(0, "../")
import gc
import time
from environs import Env
from pytz import utc
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from database.seeding import Seeding
from predictors.predictor import Predictor
from stocks_refresher.stocks_refresher import StocksRefresher

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
    StocksRefresher().refresh,
    "interval",
    minutes=10,
    id="stocks_refresher_job",
    replace_existing=True,
)

scheduler.add_job(
    Predictor().predict_and_save,
    "interval",
    minutes=120,
    id="prediction_job",
    replace_existing=True,
    next_run_time=datetime.now(),
)

scheduler.start()

while True:
    time.sleep(10)
    gc.collect()
