import copy
import pickle
from typing import Union

import pandas as pd

from sam_ml.data import Embeddings_builder, Sampler, Scaler, Selector

from .main_classifier import Classifier
from .RandomForestClassifier import RFC


class Pipe(Classifier):
    """ pipeline Wrapper class """

    def __init__(self, vectorizer: Embeddings_builder = None, scaler: Scaler = None, selector: Selector = None, sampler: Sampler = None, model: Classifier = RFC(), model_name: str = "pipe"):
        """
        @params:
            ...
            model_name: name of the model
        """
        super().__init__(model.model, model_name, model.model_type, model.grid)
        self.vectorizer = vectorizer
        self.vectorizer_dict: dict[str, Embeddings_builder] = {}
        self.scaler = scaler
        self.selector = selector
        self.sampler = sampler

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, console_out: bool = True) -> tuple[float, str]:
        if self.vectorizer is not None:
            x_train = self.__auto_vectorizing(x_train, train_on=True)
        if self.scaler is not None:
            x_train = self.scaler.scale(x_train, train_on=True)
        if self.selector is not None:
            x_train = self.selector.select(x_train, y_train, train_on=True)
        if self.sampler is not None:
            x_train, y_train = self.sampler.sample(x_train, y_train)
        return super().train(x_train, y_train, console_out)

    def __auto_vectorizing(self, X: pd.DataFrame, train_on: bool = True) -> pd.DataFrame:
        # detect string columns and create a vectorizer for each
        if train_on:
            X = X.convert_dtypes()
            string_columns = list(X.select_dtypes(include="string").columns)
            self._string_columns = string_columns
            self.vectorizer_dict = dict(zip(self._string_columns, [copy.deepcopy(self.vectorizer) for i in range(len(string_columns))]))

        for col in self._string_columns:
            X = pd.concat([X, self.vectorizer_dict[col].vectorize(X[col], train_on=train_on)], axis=1)
        X_vec = X.drop(columns=self._string_columns)

        return X_vec

    def predict(self, x_test: pd.DataFrame) -> list:
        if self.vectorizer is not None:
            x_test = self.__auto_vectorizing(x_test, train_on=False)
        if self.scaler is not None:
            x_test = self.scaler.scale(x_test, train_on=False)
        if self.selector is not None:
            x_test = self.selector.select(x_test, train_on=False)
        return super().predict(x_test)

    def cross_validation(self, X: pd.DataFrame, y: pd.Series, cv_num: int = 3, avg: str = "macro", pos_label: Union[int, str] = -1, return_estimator: bool = False, console_out: bool = True, return_as_dict: bool = False, secondary_scoring: str = None, strength: int = 3) -> Union[dict[str, float], pd.DataFrame]:
        pass

    def cross_validation_small_data(self, X: pd.DataFrame, y: pd.Series, sampling: str = "ros", vectorizer: str = "tfidf", scaler: str = "standard", avg: str = "macro", pos_label: Union[int, str] = -1, leave_loadbar: bool = True, console_out: bool = True, secondary_scoring: str = None, strength: int = 3) -> dict[str, float]:
        pass

    def gridsearch(self, x_train: pd.DataFrame, y_train: pd.Series, grid: dict = None, scoring: str = "accuracy", avg: str = "macro", pos_label: Union[int, str] = 1, n_split_num: int = 10, n_repeats_num: int = 3, verbose: int = 0, rand_search: bool = True, n_iter_num: int = 75, console_out: bool = True, train_afterwards: bool = True, secondary_scoring: str = None, strength: int = 3):
        pass
