import copy

import pandas as pd

from sam_ml.config import setup_logger
from sam_ml.data.preprocessing import (
    Embeddings_builder,
    Sampler,
    SamplerPipeline,
    Scaler,
    Selector,
)

from ..main_classifier import Classifier
from .RandomForestClassifier import RFC

logger = setup_logger(__name__)


class Pipeline(Classifier):
    """ classifier pipeline class """

    def __init__(self, vectorizer: str | Embeddings_builder | None = None, scaler: str | Scaler | None = None, selector: str | tuple[str, int] | Selector | None = None, sampler: str | Sampler | SamplerPipeline | None = None, model: Classifier = RFC(), model_name: str = "pipe"):
        """
        @params:
            vectorizer: type of "data.embeddings.Embeddings_builder" or Embeddings_builder class object for automatic string column vectorizing (None for no vectorizing)
            scaler: type of "data.scaler.Scaler" or Scaler class object for scaling the data (None for no scaling)
            selector: type of "data.feature_selection.Selector" or Selector class object for feature selection (None for no selecting)
            sampling: type of "data.sampling.Sampler" or Sampler class object for sampling the train data (None for no sampling)
            model: Classifier class object
            model_name: name of the model
        """
        if issubclass(type(model), Classifier):
            super().__init__(model_object=model.model, model_name=model_name, model_type=model.model_type, grid=model.grid)

            # Inherit methods and attributes from model
            for attribute_name in dir(model):
                attribute_value = getattr(model, attribute_name)

                # Check if the attribute is a method or a variable (excluding private attributes)
                if callable(attribute_value) and not attribute_name.startswith("__"):
                    if not hasattr(self, attribute_name):
                        setattr(self, attribute_name, attribute_value)
                elif not attribute_name.startswith("__"):
                    if not hasattr(self, attribute_name):
                        self.__dict__[attribute_name] = attribute_value

            self.__classifier = model
        else:
            raise ValueError(f"wrong input '{model}' for model")

        if vectorizer in Embeddings_builder.params()["vec"]:
            self.vectorizer = Embeddings_builder(algorithm=vectorizer)
        elif type(vectorizer) == Embeddings_builder or vectorizer is None:
            self.vectorizer = vectorizer
        else:
            raise ValueError(f"wrong input '{vectorizer}' for vectorizer")

        if scaler in Scaler.params()["scaler"]:
            self.scaler = Scaler(algorithm=scaler)
        elif type(scaler) == Scaler or scaler is None:
            self.scaler = scaler
        else:
            raise ValueError(f"wrong input '{scaler}' for scaler")

        if selector in Selector.params()["algorithm"]:
            self.selector = Selector(algorithm=selector)
        elif type(selector) == tuple and len(selector) == 2:
            if selector[0] in Selector.params()["algorithm"] and type(selector[1])==int:
                if selector[1] > 0:
                    self.selector = Selector(algorithm=selector[0], num_features=selector[1])
                else:
                    raise ValueError(f"wrong input '{selector}' for selector -> integer in tuple has to be greater 0")
            else:
                raise ValueError(f"wrong input '{selector}' for selector -> tuple incorrect")
        elif type(selector) == Selector or selector is None:
            self.selector = selector
        else:
            raise ValueError(f"wrong input '{selector}' for selector")

        if sampler in Sampler.params()["algorithm"]:
            self.sampler = Sampler(algorithm=sampler)
        elif type(sampler) ==str and SamplerPipeline.check_is_valid_algorithm(sampler):
            self.sampler = SamplerPipeline(algorithm=sampler)
        elif type(sampler) in (Sampler, SamplerPipeline) or sampler is None:
            self.sampler = sampler
        else:
            raise ValueError(f"wrong input '{sampler}' for sampler")

        self.vectorizer_dict: dict[str, Embeddings_builder] = {}

        # keep track if model was trained for warm_start
        self._data_classes_trained: bool = False

    def __repr__(self) -> str:
        params: str = ""
        for step in self.steps:
            params += step[0]+"="+step[1].__str__()+", "

        params += f"model_name='{self.model_name}'"

        return f"Pipeline({params})"

    @property
    def steps(self) -> list[tuple[str, any]]:
        return [("vectorizer", self.vectorizer), ("scaler", self.scaler), ("selector", self.selector), ("sampler", self.sampler), ("model", self.__classifier)]
    
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
        self._data_classes_trained = True
        return X, y

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        x_train_pre, y_train_pre = self.__data_prepare(x_train, y_train, train_on=True)
        self.feature_names = list(x_train_pre.columns)
        return super().fit(x_train_pre, y_train_pre, **kwargs)
    
    def fit_warm_start(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        x_train_pre, y_train_pre = self.__data_prepare(x_train, y_train, train_on = not self._data_classes_trained)
        self.feature_names = list(x_train_pre.columns)
        return super().fit(x_train_pre, y_train_pre, **kwargs)

    def predict(self, x_test: pd.DataFrame) -> list:
        x_test_pre, _ = self.__data_prepare(x_test, None, train_on=False)
        return super().predict(x_test_pre)

    def get_params(self, deep: bool = True) -> dict[str, any]:
        return dict(self.steps)
