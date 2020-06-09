import sys

sys.path.insert(0, "../")
from environs import Env
from datetime import date
import json
from utilities.mongo_connection import get_db_connection

# from predictors.lstm import lstm_predict

import os

env = Env()
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
env.read_env(path=env_path)
api_key = env("alphavantage_api_key")


class Predictor:
    def predict_and_save(self):
        symbol = "TSLA"
        print(f"Running prediction for {symbol}")

        # model_loss, lstm_forecast = lstm_predict(symbol)

        title = "Stock Prediction for {} ".format(symbol)

        self.save_model_in_db(
            symbol=symbol,
            title="some new prediction",
            model_accuracy="0.0",
            model_prediction="0.0",
        )

        # self.save_model_in_db(
        #     symbol,
        #     title,
        #     # stock_history,
        #     lstm_forecast,
        #     # train_mean_error,
        #     # test_mean_error,
        # )

    def save_model_in_db(
        self, symbol, title, model_accuracy, model_prediction,
    ):
        conn = get_db_connection()
        stocksdb = conn["stocksdb"]
        modelCollection = stocksdb["models"]

        dbModel = {
            "symbol": symbol,
            "title": title,
            "model_accuracy": model_accuracy,
            "model_prediction": model_prediction,
            "date_created": date.today().strftime("%Y-%m-%d"),
        }

        modelCollection.find_one_and_replace({"symbol": symbol}, dbModel, upsert=True)
