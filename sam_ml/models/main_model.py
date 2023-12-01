import pickle
import time
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd

from sam_ml.config import setup_logger

logger = setup_logger(__name__)

class Model:
    """ Model parent class {abstract} """

    def __init__(self, model_object, model_name: str, model_type: str):
        """
        Parameters
        ----------
        model_object : model object
            model with 'fit', 'predict', 'set_params', and 'get_params' method (see sklearn API)
        model_name : str
            name of the model
        model_type : str
            kind of estimator (e.g. 'RFC' for RandomForestClassifier)
        """
        self._model = model_object
        self._model_name = model_name
        self._model_type = model_type
        self._train_score: float = None
        self._train_time: str = None
        self._feature_names: list[str] = []

    def __repr__(self) -> str:
        return f"Model(model_object={self.model.__str__()}, model_name='{self.model_name}', model_type='{self.model_type}')"
    
    @property
    def model(self):
        """
        Returns
        -------
        model : model object
            model with 'fit', 'predict', 'set_params', and 'get_params' method (see sklearn API)
        """
        return self._model
    
    @property
    def model_name(self) -> str:
        """
        Returns
        -------
        model_name : str
            name of the model
        """
        return self._model_name
    
    @property
    def model_type(self) -> str:
        """
        Returns
        -------
        model_type : str
            kind of estimator (e.g. 'RFC' for RandomForestClassifier)
        """
        return self._model_type
    
    @property
    def train_score(self) -> float:
        """
        Returns
        -------
        train_score : float
            train score value
        """
        return self._train_score
    
    @property
    def train_time(self) -> str:
        """
        Returns
        -------
        train_time : str
            train time in format: "0:00:00" (hours:minutes:seconds)
        """
        return self._train_time
    
    @property
    def feature_names(self) -> list[str]:
        """
        Returns
        -------
        feature_names : list[str]
            names of all the features that the model saw during training. Is empty if model was not fitted yet.
        """
        return self._feature_names

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, console_out: bool = True, **kwargs) -> tuple[float, str]:
        """
        Function to train the model

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train model
        console_out : bool, \
                default=True
            shall the score and time be printed out
        **kwargs:
            additional parameters from child-class

        Returns
        -------
        train_score : float 
            train score value
        train_time : str
            train time in format: "0:00:00" (hours:minutes:seconds)
        """
        logger.debug(f"training {self.model_name} - started")

        start_time = time.time()
        self.fit(x_train, y_train)
        end_time = time.time()
        self._train_score = self.evaluate_score(x_train, y_train, **kwargs)
        self._train_time = str(timedelta(seconds=int(end_time-start_time)))

        if console_out:
            print(f"Train score: {self.train_score} - Train time: {self.train_time}")
            
        logger.debug(f"training {self.model_name} - finished")

        return self.train_score, self.train_time
    
    def train_warm_start(self, x_train: pd.DataFrame, y_train: pd.Series, console_out: bool = True, **kwargs) -> tuple[float, str]:
        """
        Function to warm_start train the model

        This function only differs for pipeline objects (with preprocessing) from the train method.
        For pipeline objects, it only traines the preprocessing steps the first time and then only uses them to preprocess.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train model
        console_out : bool, \
                default=True
            shall the score and time be printed out
        **kwargs:
            additional parameters from child-class

        Returns
        -------
        train_score : float 
            train score value
        train_time : str
            train time in format: "0:00:00" (hours:minutes:seconds)
        """
        logger.debug(f"training {self.model_name} - started")

        start_time = time.time()
        self.fit_warm_start(x_train, y_train)
        end_time = time.time()
        self._train_score = self.evaluate_score(x_train, y_train, **kwargs)
        self._train_time = str(timedelta(seconds=int(end_time-start_time)))

        if console_out:
            print(f"Train score: {self.train_score} - Train time: {self.train_time}")
            
        logger.debug(f"training {self.model_name} - finished")

        return self.train_score, self.train_time

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Function to fit the model

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train model
        **kwargs:
            additional parameters from child-class

        Returns
        -------
        self : estimator instance
            Estimator instance
        """
        self._feature_names = list(x_train.columns)
        self.model.fit(x_train, y_train, **kwargs)
        return self
    
    def fit_warm_start(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Function to warm_start fit the model

        This function only differs for pipeline objects (with preprocessing) from the train method.
        For pipeline objects, it only traines the preprocessing steps the first time and then only uses them to preprocess.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train model
        **kwargs:
            additional parameters from child-class

        Returns
        -------
        self : estimator instance
            Estimator instance
        """
        self._feature_names = list(x_train.columns)
        self.model.fit(x_train, y_train, **kwargs)
        return self

    def predict(self, x_test: pd.DataFrame) -> list:
        """
        Function to predict with predict-method from model object

        Parameters
        ----------
        x_test : pd.DataFrame
            Data for prediction

        Returns
        -------
        prediction : list
            list with predicted class numbers for data
        """
        return list(self.model.predict(x_test))
    
    def predict_proba(self, x_test: pd.DataFrame) -> np.ndarray:
        """
        Function to predict with predict_proba-method from model object

        Parameters
        ----------
        x_test : pd.DataFrame
            Data for prediction

        Returns
        -------
        prediction : np.ndarray
            np.ndarray with probability for every class per datapoint
        """
        try:
            return self.model.predict_proba(x_test)
        except:
            raise NotImplementedError(f"predict_proba for {self.model_name} is not implemented")

    def get_params(self, deep: bool = True) -> dict:
        """
        Function to get the parameter from the model object

        Parameters
        ----------
        deep : bool, \
                default=True
            If True, will return the parameters for this estimator and contained sub-objects that are estimators

        Returns
        -------
        params: dict
            parameter names mapped to their values
        """
        return self.model.get_params(deep)

    def set_params(self, **params):
        """
        Function to set the parameter of the model object

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self : estimator instance
            Estimator instance
        """
        self.model.set_params(**params)
        return self

    def evaluate_score(self, x_test: pd.DataFrame, y_test: pd.Series, **kwargs) -> float:
        """
        Function to create a score with score-function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data for evaluating the model
        **kwargs:
            additional parameters from child-class

        Returns
        -------
        score : float
            metrics score value
        """
        score = self.model.score(x_test, y_test)
        return score
    
    def get_deepcopy(self):
        """
        Function to create a deepcopy of object

        Returns
        -------
        self : estimator instance
            deepcopy of estimator instance
        """
        return deepcopy(self)

    def save_model(self, path: str, only_estimator: bool = False):
        """ 
        Function to pickle and save the class object
        
        Parameters
        ----------
        path : str
            path to save the model with suffix '.pkl'
        only_estimator : bool, \
                default=False
            If True, only the estimator of the class object will be saved
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
        """ 
        Function to load a pickled model class object
        
        Parameters
        ----------
        path : str
            path to save the model with suffix '.pkl'

        Returns
        -------
        model : estimator instance
            estimator instance
        """
        logger.debug("loading model - started")
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.debug("loading model - finished")
        return model
