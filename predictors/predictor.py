# %%
import sys

sys.path.insert(0, "..")
from environs import Env
from utilities.get_abs_path import get_abs_path

# from predictors.lstm import lstm_predict

env = Env()


class Predictor:
    def predict_and_save(self):
        symbol = "TSLA"
        print(f"Running prediction for {symbol}")

        filename = get_abs_path(__file__, "./lstm_predictor.py")

        with open(filename, "rb") as source_file:
            code = compile(source_file.read(), filename, "exec")
        exec(code)


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict_and_save()

