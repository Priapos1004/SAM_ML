import pickle
import pandas as pd
import logging

class Model:

    def __init__(self, model_object = None):
        self.model = model_object

    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        logging.debug("training started...")
        self.model.fit(x_train, y_train)
        self.feature_names = x_train.columns
        logging.debug("... training finished")

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series, console_out: bool = True):
        pass

    def save(self, path: str):
        logging.debug("saving started...")
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logging.debug("... model saved")

    def load(self, path: str):
        logging.debug("loading model...")
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logging.debug("... model loaded")
    