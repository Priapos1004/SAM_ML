import copy
from typing import Union

import pandas as pd

from sam_ml.data import Embeddings_builder, Sampler, Scaler, Selector

from .main_classifier import Classifier
from .RandomForestClassifier import RFC


class Pipe(Classifier):
    """ pipeline Wrapper class """

    def __init__(self, vectorizer: Union[str, Embeddings_builder] = None, scaler: Union[str, Scaler] = None, selector: Union[str, Selector] = None, sampler: Union[str, Sampler] = None, model: Union[tuple, Classifier] = RFC(), model_name: str = "pipe"):
        """
        @params:
            ...
            model_name: name of the model
        """
        self._classifier: tuple

        if issubclass(type(model), Classifier):
            super().__init__(model.model, model_name, model.model_type, model.grid, is_pipeline = True)
            self._classifier = (model.model, model.model_type, model.grid)
        else:
            super().__init__(model[0], model_name, model[1], model[2], is_pipeline = True)
            self._classifier = model

        if vectorizer in Embeddings_builder.params()["vec"]:
            self.vectorizer = Embeddings_builder(vec=vectorizer)
        else:
            self.vectorizer = vectorizer

        if scaler in Scaler.params()["scaler"]:
            self.scaler = Scaler(scaler=scaler)
        else:
            self.scaler = scaler

        if selector in Selector.params()["algorithm"]:
            self.selector = Selector(algorithm=selector)
        else:
            self.selector = selector

        if sampler in Sampler.params()["algorithm"]:
            self.sampler = Sampler(algorithm=sampler)
        else:
            self.sampler = sampler

        self.vectorizer_dict: dict[str, Embeddings_builder] = {}

    @property
    def steps(self):
        return [("vectorizer", self.vectorizer), ("scaler", self.scaler), ("selector", self.selector), ("sampler", self.sampler), ("model", self._classifier)]

    @property
    def grid(self):
        pre_grid = {}
        vectorizer_grid = {f"vectorizer__{k}": v for k, v in self.vectorizer._grid.items()}
        pre_grid.update(vectorizer_grid)
        scaler_grid = {f"scaler__{k}": v for k, v in self.scaler._grid.items()}
        pre_grid.update(scaler_grid)
        selector_grid = {f"selector__{k}": v for k, v in self.selector._grid.items()}
        pre_grid.update(selector_grid)
        sampler_grid = {f"sampler__{k}": v for k, v in self.sampler._grid.items()}
        pre_grid.update(sampler_grid)
        model_grid = {f"model__{k}": v for k, v in self._grid.items()}
        pre_grid.update(model_grid)
        return pre_grid

    def update_grid(self, **kwargs):
        vectorizer_params = dict([[i.split("__")[1], kwargs[i]] for i in list(kwargs.keys()) if i.split("__")[0] == "vectorizer"])
        self.vectorizer._grid.update(vectorizer_params)
        scaler_params = dict([[i.split("__")[1], kwargs[i]] for i in list(kwargs.keys()) if i.split("__")[0] == "scaler"])
        self.scaler._grid.update(scaler_params)
        selector_params = dict([[i.split("__")[1], kwargs[i]] for i in list(kwargs.keys()) if i.split("__")[0] == "selector"])
        self.selector._grid.update(selector_params)
        sampler_params = dict([[i.split("__")[1], kwargs[i]] for i in list(kwargs.keys()) if i.split("__")[0] == "sampler"])
        self.sampler._grid.update(sampler_params)
        model_params = dict([[i.split("__")[1], kwargs[i]] for i in list(kwargs.keys()) if i.split("__")[0] == "model"])
        super().update_grid(**model_params)
    
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

    def __data_prepare(self, X: pd.DataFrame, y: pd.Series, train_on: bool = True):
        if self.vectorizer is not None:
            X = self.__auto_vectorizing(X, train_on=train_on)
        if self.scaler is not None:
            X = self.scaler.scale(X, train_on=train_on)
        if self.selector is not None:
            X = self.selector.select(X, y, train_on=train_on)
        if self.sampler is not None and train_on:
            X, y = self.sampler.sample(X, y)
        return X, y

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, console_out: bool = True) -> tuple[float, str]:
        x_train_pre, y_train_pre = self.__data_prepare(x_train, y_train, train_on=True)
        return super().train(x_train_pre, y_train_pre, console_out)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train_pre, y_train_pre = self.__data_prepare(x_train, y_train, train_on=True)
        return super().fit(x_train_pre, y_train_pre)

    def predict(self, x_test: pd.DataFrame) -> list:
        x_test_pre, _ = self.__data_prepare(x_test, None, train_on=False)
        return super().predict(x_test_pre)

    def get_params(self, deep: bool = True):
        return dict(self.steps)

    def set_params(self, **params):
        params = dict(**params)
        vec_params = dict([[i.split("__")[1], params[i]] for i in list(params.keys()) if i.split("__")[0] == "vectorizer"])
        self.vectorizer.vectorizer.set_params(**vec_params)
        scaler_params = dict([[i.split("__")[1], params[i]] for i in list(params.keys()) if i.split("__")[0] == "scaler"])
        self.scaler.scaler.set_params(**scaler_params)
        selector_params = dict([[i.split("__")[1], params[i]] for i in list(params.keys()) if i.split("__")[0] == "selector"])
        self.selector.selector.set_params(**selector_params)
        sampler_params = dict([[i.split("__")[1], params[i]] for i in list(params.keys()) if i.split("__")[0] == "sampler"])
        self.sampler.sampler.set_params(**sampler_params)
        model_params = dict([[i.split("__")[1], params[i]] for i in list(params.keys()) if i.split("__")[0] == "model"])
        super().set_params(**model_params)
        return self

    def cross_validation_small_data(self, X: pd.DataFrame, y: pd.Series, sampling: str = "ros", vectorizer: str = "tfidf", scaler: str = "standard", avg: str = "macro", pos_label: Union[int, str] = -1, leave_loadbar: bool = True, console_out: bool = True, secondary_scoring: str = None, strength: int = 3) -> dict[str, float]:
        pass
