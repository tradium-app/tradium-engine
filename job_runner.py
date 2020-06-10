import sys

sys.path.insert(0, "../")
import gc
import time
from environs import Env
from utilities.mongo_connection import get_db_connection
from pytz import utc
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from predictors.predictor import Predictor

env = Env()
env.read_env()

jobstores = {
    "default": MongoDBJobStore(database=env("MONGO_DB"), client=get_db_connection()),
}

executors = {"default": ThreadPoolExecutor(20), "processpool": ProcessPoolExecutor(5)}

job_defaults = {"coalesce": False, "max_instances": 3}
scheduler = BackgroundScheduler(
    jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone=utc
)


job = scheduler.add_job(
    Predictor().predict_and_save,
    "interval",
    minutes=10,
    id="35fd79fa7ee7484696176b919ef6b431",
    replace_existing=True,
)

scheduler.start()

while True:
    time.sleep(10)
    gc.collect()
