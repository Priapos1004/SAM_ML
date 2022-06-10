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
        self.train_score = None
        self.train_time = None
        self.test_score = None

    def train(
        self, x_train: pd.DataFrame, y_train: pd.Series, console_out: bool = True
    ) -> tuple[float, str]:
        logging.debug("training started...")
        start_time = time.time()
        self.model.fit(x_train, y_train)
        end_time = time.time()
        self.feature_names = x_train.columns
        self.train_score = self.model.score(x_train, y_train)
        self.train_time = str(timedelta(seconds=int(end_time-start_time)))
        if console_out:
            print("Train score: ", self.train_score, " - Train time: ", self.train_time)
        logging.debug("... training finished")
        return self.train_score, self.train_time

    def evaluate(
        self, x_test: pd.DataFrame, y_test: pd.Series, console_out: bool = True
    ) -> float:
        self.test_score = self.model.score(x_test, y_test)
        if console_out:
            print("Test score: ", self.test_score)
        return self.test_score

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
