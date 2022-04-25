import logging
import pickle
import time
from datetime import timedelta

import pandas as pd


class Model:
    def __init__(self, model_object=None, model_name="model"):
        self.model = model_object
        self.model_name = model_name
        self.model_type = "Model"

    def train(
        self, x_train: pd.DataFrame, y_train: pd.Series, console_out: bool = True
    ):
        logging.debug("training started...")
        start_time = time.time()
        self.model.fit(x_train, y_train)
        end_time = time.time()
        self.feature_names = x_train.columns
        if console_out:
            print("Train score: ", self.model.score(x_train, y_train), " - Train time: ", str(timedelta(seconds=int(end_time-start_time))))
        logging.debug("... training finished")

    def evaluate(
        self, x_test: pd.DataFrame, y_test: pd.Series, console_out: bool = True
    ) -> float:
        score = self.model.score(x_test, y_test)
        if console_out:
            print("Test score: ", score)
        return score

    def save(self, path: str):
        logging.debug("saving started...")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logging.debug("... model saved")

    def load(self, path: str):
        logging.debug("loading model...")
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        logging.debug("... model loaded")
