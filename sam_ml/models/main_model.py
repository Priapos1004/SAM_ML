import pickle
import time
from datetime import timedelta
from typing import Union

import pandas as pd


class Model:
    """ Model parent class """

    def __init__(self, model_object = None, model_name: str = "model", model_type: str = "Model"):
        """
        @params:
            model_object: model with 'fit' and 'predict' method
            model_name: name of the model
            model_type: kind of estimator  (e.g. 'RFC' for RandomForestClassifier)
        """
        self.model = model_object
        self.model_name = model_name
        self.model_type = model_type
        self.trained: bool = False
        self.train_score: float = None
        self.train_time: str = None
        self.test_score: Union[float, dict[str, float]] = None
        self.feature_names: list = None

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, console_out: bool = True) -> tuple[float, str]:
        if console_out:
            print("started training...")

        start_time = time.time()
        self.model.fit(x_train, y_train)
        end_time = time.time()
        self.feature_names = list(x_train.columns)
        self.train_score = self.model.score(x_train, y_train)
        self.train_time = str(timedelta(seconds=int(end_time-start_time)))

        if console_out:
            print("Train score: ", self.train_score, " - Train time: ", self.train_time)

        self.trained = True
        return self.train_score, self.train_time

    def predict(self, x_test: pd.DataFrame) -> list:
        return list(self.model.predict(x_test))

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series, console_out: bool = True) -> float:
        self.test_score = self.model.score(x_test, y_test)
        if console_out:
            print("Test score: ", self.test_score)
        return self.test_score

    def save_model(self, path: str, only_estimator: bool = False):
        """ 
        function to pickle and save the Class object 
        
        @params:
            path: path to save the model with suffix '.pkl'
            only_estimator: if True, only the estimator of the class object will be saved
        """
        print("saving started...")
        with open(path, "wb") as f:
            if only_estimator:
                pickle.dump(self.model, f)
            else:
                pickle.dump(self, f)
        print("... model saved")

    @staticmethod
    def load_model(path: str):
        """ function to load a pickled Model class object """
        print("loading model...")
        with open(path, "rb") as f:
            model = pickle.load(f)
        print("... model loaded")
        return Model(model, model.model_name, model.model_type)
