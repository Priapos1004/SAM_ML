import pickle
import time
from datetime import timedelta
from typing import Union

import pandas as pd

from sam_ml.config import setup_logger

logger = setup_logger(__name__)

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

    def __repr__(self) -> str:
        return f"Model(model_object={self.model.__str__()}, model_name='{self.model_name}', model_type='{self.model_type}')"

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, console_out: bool = True) -> tuple[float, str]:
        """
        @return:
            tuple of train score and train time
        """
        logger.debug(f"training {self.model_name} - started")

        start_time = time.time()
        self.model.fit(x_train, y_train)
        end_time = time.time()
        self.feature_names = list(x_train.columns)
        self.train_score = self.model.score(x_train, y_train)
        self.train_time = str(timedelta(seconds=int(end_time-start_time)))

        if console_out:
            print("Train score: ", self.train_score, " - Train time: ", self.train_time)
            
        logger.debug(f"training {self.model_name} - finished")

        self.trained = True

        return self.train_score, self.train_time

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test: pd.DataFrame) -> list:
        """
        @return:
            list with predictions
        """
        return list(self.model.predict(x_test))

    def get_params(self, deep: bool = True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

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
        logger.debug(f"saving {self.model_name} - started")
        with open(path, "wb") as f:
            if only_estimator:
                pickle.dump(self.model, f)
            else:
                pickle.dump(self, f)
        logger.debug(f"saving {self.model_name} - finished")

    @staticmethod
    def load_model(path: str):
        """ function to load a pickled model class object """
        logger.debug("loading model - started")
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.debug("loading model - finished")
        return model
