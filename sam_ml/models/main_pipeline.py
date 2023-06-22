import copy
from typing import Union

import pandas as pd

from sam_ml.config import setup_logger
from sam_ml.data import Embeddings_builder, Sampler, Scaler, Selector

from .main_classifier import Classifier
from .RandomForestClassifier import RFC

logger = setup_logger(__name__)


class Pipeline(Classifier):
    """ classifier pipeline class """

    def __init__(self, vectorizer: Union[str, Embeddings_builder] = None, scaler: Union[str, Scaler] = None, selector: Union[str, Selector] = None, sampler: Union[str, Sampler] = None, model: Union[tuple, Classifier] = RFC(), model_name: str = "pipe"):
        """
        @params:
            vectorizer: type of "data.embeddings.Embeddings_builder" or Embeddings_builder class object for automatic string column vectorizing (None for no vectorizing)
            scaler: type of "data.scaler.Scaler" or Scaler class object for scaling the data (None for no scaling)
            selector: type of "data.feature_selection.Selector" or Selector class object for feature selection (None for no selecting)
            sampling: type of "data.sampling.Sampler" or Sampler class object for sampling the train data (None for no sampling)
            model: Classifier class object or tuple (sklearn model, model_type, hyperparameter grid)
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
        elif type(vectorizer) == Embeddings_builder or vectorizer is None:
            self.vectorizer = vectorizer
        else:
            logger.error(f"wrong input '{vectorizer}' for vectorizer -> vectorizer = None")
            self.vectorizer = None

        if scaler in Scaler.params()["scaler"]:
            self.scaler = Scaler(scaler=scaler)
        elif type(scaler) == Scaler or scaler is None:
            self.scaler = scaler
        else:
            logger.error(f"wrong input '{scaler}' for scaler -> scaler = None")
            self.scaler = None

        if selector in Selector.params()["algorithm"]:
            self.selector = Selector(algorithm=selector)
        elif type(selector) == Selector or selector is None:
            self.selector = selector
        else:
            logger.error(f"wrong input '{selector}' for selector -> selector = None")
            self.selector = None

        if sampler in Sampler.params()["algorithm"]:
            self.sampler = Sampler(algorithm=sampler)
        elif type(sampler) == Sampler or sampler is None:
            self.sampler = sampler
        else:
            logger.error(f"wrong input '{sampler}' for sampler -> sampler = None")
            self.sampler = None

        # check for incompatible sampler-model combination
        if self.sampler is not None:
            sampling_problems = ["QDA", "LDA", "LR", "MLPC", "LSVC"]
            if self.sampler.algorithm == "SMOTE" and self.model_type in sampling_problems:
                logger.warning(self.model_type+" does not work with sampling='SMOTE' --> going on with sampling='ros'")
                self.sampler = Sampler(algorithm="ros")
            elif self.sampler.algorithm in ("nm", "tl") and self.model_type in sampling_problems:
                logger.warning(self.model_type+f" does not work with sampling='{self.sampler.algorithm}' --> going on with sampling='rus'")
                self.sampler = Sampler(algorithm="rus")

        self.vectorizer_dict: dict[str, Embeddings_builder] = {}

    def __repr__(self) -> str:
        params: str = ""
        data_steps = ("vectorizer", self.vectorizer), ("scaler", self.scaler), ("selector", self.selector), ("sampler", self.sampler)
        for step in data_steps:
            params += step[0]+"="+step[1].__str__()+", "

        params += f"model={self.model.__str__()}, "
        params += f"model_name='{self.model_name}'"

        return f"Pipeline({params})"

    @property
    def steps(self) -> list[tuple[str, any]]:
        return [("vectorizer", self.vectorizer), ("scaler", self.scaler), ("selector", self.selector), ("sampler", self.sampler), ("model", self._classifier)]

    # @property
    # def grid(self) -> dict[str, list]:
    #     pre_grid = {}
    #     if self.vectorizer is not None:
    #         vectorizer_grid = {f"vectorizer__{k}": v for k, v in self.vectorizer._grid.items()}
    #         pre_grid.update(vectorizer_grid)
    #     if self.scaler is not None:
    #         scaler_grid = {f"scaler__{k}": v for k, v in self.scaler._grid.items()}
    #         pre_grid.update(scaler_grid)
    #     if self.selector is not None:
    #         selector_grid = {f"selector__{k}": v for k, v in self.selector._grid.items()}
    #         pre_grid.update(selector_grid)
    #     if self.sampler is not None:
    #         sampler_grid = {f"sampler__{k}": v for k, v in self.sampler._grid.items()}
    #         pre_grid.update(sampler_grid)
    #     model_grid = {f"model__{k}": v for k, v in self._grid.items()}
    #     pre_grid.update(model_grid)
    #     return pre_grid

    # def update_grid(self, **kwargs) -> None:
    #     if self.vectorizer is not None:
    #         vectorizer_params = dict([[i.split("__")[1], kwargs[i]] for i in list(kwargs.keys()) if i.split("__")[0] == "vectorizer"])
    #         self.vectorizer._grid.update(vectorizer_params)
    #     if self.scaler is not None:
    #         scaler_params = dict([[i.split("__")[1], kwargs[i]] for i in list(kwargs.keys()) if i.split("__")[0] == "scaler"])
    #         self.scaler._grid.update(scaler_params)
    #     if self.selector is not None:
    #         selector_params = dict([[i.split("__")[1], kwargs[i]] for i in list(kwargs.keys()) if i.split("__")[0] == "selector"])
    #         self.selector._grid.update(selector_params)
    #     if self.sampler is not None:
    #         sampler_params = dict([[i.split("__")[1], kwargs[i]] for i in list(kwargs.keys()) if i.split("__")[0] == "sampler"])
    #         self.sampler._grid.update(sampler_params)
    #     model_params = dict([[i.split("__")[1], kwargs[i]] for i in list(kwargs.keys()) if i.split("__")[0] == "model"])
    #     super().update_grid(**model_params)
    
    def __auto_vectorizing(self, X: pd.DataFrame, train_on: bool = True) -> pd.DataFrame:
        """ detects string columns, creates a vectorizer for each, and vectorizes them """
        if train_on:
            X = X.convert_dtypes()
            string_columns = list(X.select_dtypes(include="string").columns)
            self._string_columns = string_columns
            self.vectorizer_dict = dict(zip(self._string_columns, [copy.deepcopy(self.vectorizer) for i in range(len(string_columns))]))

        for col in self._string_columns:
            X = pd.concat([X, self.vectorizer_dict[col].vectorize(X[col], train_on=train_on)], axis=1)
        X_vec = X.drop(columns=self._string_columns)

        return X_vec

    def __data_prepare(self, X: pd.DataFrame, y: pd.Series, train_on: bool = True) -> tuple[pd.DataFrame, pd.Series]:
        """ runs data class objects on data to prepare them for the model """
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

    def get_params(self, deep: bool = True) -> dict[str, any]:
        return dict(self.steps)

    # def set_params(self, **params):
    #     if self.vectorizer is not None:
    #         vec_params = dict([[i.split("__")[1], params[i]] for i in list(params.keys()) if i.split("__")[0] == "vectorizer"])
    #         self.vectorizer.set_params(**vec_params)
    #     if self.scaler is not None:
    #         scaler_params = dict([[i.split("__")[1], params[i]] for i in list(params.keys()) if i.split("__")[0] == "scaler"])
    #         self.scaler.set_params(**scaler_params)
    #     if self.selector is not None:
    #         selector_params = dict([[i.split("__")[1], params[i]] for i in list(params.keys()) if i.split("__")[0] == "selector"])
    #         self.selector.set_params(**selector_params)
    #     if self.sampler is not None:
    #         sampler_params = dict([[i.split("__")[1], params[i]] for i in list(params.keys()) if i.split("__")[0] == "sampler"])
    #         self.sampler.set_params(**sampler_params)
    #     model_params = dict([[i.split("__")[1], params[i]] for i in list(params.keys()) if i.split("__")[0] == "model"])
    #     super().set_params(**model_params)
    #     return self
