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
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from database.seeding import Seeding
from predictors.predictor import Predictor
from stocks_refresher.stocks_refresher import StocksRefresher
from transaction_executer.transaction_executer import TransactionExecuter

env = Env()
env.read_env()
DATABASE_URL = env("DATABASE_URL")

Seeding().initialize()

jobstores = {
    "default": SQLAlchemyJobStore(url=DATABASE_URL, tablename="apscheduler_jobs"),
}

executors = {"default": ThreadPoolExecutor(20), "processpool": ProcessPoolExecutor(5)}

job_defaults = {"coalesce": False, "max_instances": 1}
scheduler = BackgroundScheduler(
    jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone=utc
)

scheduler.add_job(
    id="stocks_refresher_job",
    func=StocksRefresher().refresh,
    trigger=CronTrigger(
        day_of_week="mon,tue,wed,thu,fri", hour="8-18", minute="*/10", timezone="est"
    ),
    replace_existing=True,
)

scheduler.add_job(
    id="build_model_job",
    func=Predictor().build_model_and_save,
    trigger=CronTrigger(
        day_of_week="mon,tue,wed,thu,fri", hour="8-18/2", timezone="est"
    ),
    replace_existing=True,
    next_run_time=datetime.now(),
)

scheduler.add_job(
    id="prediction_job",
    func=Predictor().predict_and_save,
    trigger=CronTrigger(
        day_of_week="mon,tue,wed,thu,fri", hour="8-18", minute="*/30", timezone="est"
    ),
    replace_existing=True,
    next_run_time=datetime.now(),
)


def on_job_completed(event):
    if event.exception:
        print(f"Job crashed: {event.job_id}")
    elif event.job_id == "prediction_job":
        TransactionExecuter().execute_prediction()
    pass


scheduler.add_listener(on_job_completed, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

scheduler.start()

while True:
    time.sleep(10)
    gc.collect()
