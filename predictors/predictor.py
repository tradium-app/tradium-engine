# %%
import sys

sys.path.insert(0, "..")
from environs import Env
from datetime import date
import json
from notification.notification_sender import Notification_Sender


# from predictors.lstm import lstm_predict

env = Env()
# api_key = env("alphavantage_api_key")


class Predictor:
    def predict_and_save(self):
        symbol = "TSLA"
        print(f"Running prediction for {symbol}")

        # model_loss, lstm_forecast = lstm_predict(symbol)

        # title = "Stock Prediction for {} ".format(symbol)
        # self.save_model_in_db(
        #     symbol=symbol,
        #     title="some new prediction",
        #     model_accuracy="0.0",
        #     model_prediction="0.0",
        # )

        # self.save_model_in_db(
        #     symbol,
        #     title,
        #     # stock_history,
        #     lstm_forecast,
        #     # train_mean_error,
        #     # test_mean_error,
        # )

        filename = "lstm_predictor.py"
        with open(filename, "rb") as source_file:
            code = compile(source_file.read(), filename, "exec")
        exec(code)

        sender = Notification_Sender()
        sender.send()

    def save_model_in_db(
        self, symbol, title, model_accuracy, model_prediction,
    ):
        # conn = get_db_connection()
        db = Env()("MONGO_DB")
        # stocksdb = conn[db]
        # modelCollection = stocksdb["models"]

        # dbModel = {
        #     "symbol": symbol,
        #     "title": title,
        #     "model_accuracy": model_accuracy,
        #     "model_prediction": model_prediction,
        #     "date_created": date.today().strftime("%Y-%m-%d"),
        # }

        # modelCollection.find_one_and_replace({"symbol": symbol}, dbModel, upsert=True)


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict_and_save()

