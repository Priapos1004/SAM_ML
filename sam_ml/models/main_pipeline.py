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

from .main_classifier import Classifier
from .main_model import Model
from .main_regressor import Regressor

logger = setup_logger(__name__)


class BasePipeline(Model):
    """ BasePipeline class """

    def __init__(self, model: Classifier | Regressor,  vectorizer: str | Embeddings_builder | None, scaler: str | Scaler | None, selector: str | tuple[str, int] | Selector | None, sampler: str | Sampler | SamplerPipeline | None, model_name: str = "pipe"):
        """
        Parameters
        ----------
        model : Classifier or Regressor class object
            Model used in pipeline (:class:`Classifier` or :class:`Regressor`)
        vectorizer : str, Embeddings_builder, or None
            object or algorithm of :class:`Embeddings_builder` class which will be used for automatic string column vectorizing (None for no vectorizing)
        scaler : str, Scaler, or None
            object or algorithm of :class:`Scaler` class for scaling the data (None for no scaling)
        selector : str, Selector, or None
            object or algorithm of :class:`Selector` class for feature selection (None for no selecting)
        sampler : str, Sampler, or None
            object or algorithm of :class:`Sampler` class for sampling the train data (None for no sampling)
        model_name : str
            name of the model
        """
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

        self.__model = model

        if vectorizer in Embeddings_builder.params()["algorithm"]:
            self.vectorizer = Embeddings_builder(algorithm=vectorizer)
        elif type(vectorizer) == Embeddings_builder or vectorizer is None:
            self.vectorizer = vectorizer
        else:
            raise ValueError(f"wrong input '{vectorizer}' for vectorizer")

        if scaler in Scaler.params()["algorithm"]:
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
        elif type(sampler) ==str:
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
        """
        Returns
        -------
        steps : list[tuple[str, any]]
            list with preprocessing + model pipeline steps as tuples
        """
        return [("vectorizer", self.vectorizer), ("scaler", self.scaler), ("selector", self.selector), ("sampler", self.sampler), ("model", self.__model)]
    
    def _auto_vectorizing(self, X: pd.DataFrame, train_on: bool) -> pd.DataFrame:
        """
        Function to detect string columns and creating a vectorizer for each, and vectorize them 
        
        Parameters
        ----------
        X : pd.DataFrame
            data to vectorize
        train_on : bool
            if data shall just be transformed (``train_on=False``) or also the vectorizer be trained before

        Returns
        -------
        X_vectorized : pd.DataFrame
            dataframe X with string columns replaced by vectorize columns
        """
        if train_on:
            X = X.convert_dtypes()
            string_columns = list(X.select_dtypes(include="string").columns)
            self._string_columns = string_columns
            self.vectorizer_dict = dict(zip(self._string_columns, [copy.deepcopy(self.vectorizer) for i in range(len(string_columns))]))

        for col in self._string_columns:
            X = pd.concat([X, self.vectorizer_dict[col].vectorize(X[col], train_on=train_on)], axis=1)
        X_vectorized = X.drop(columns=self._string_columns)

        return X_vectorized

    def data_prepare(self, X: pd.DataFrame, y: pd.Series, train_on: bool = True) -> tuple[pd.DataFrame, pd.Series]:
        """ 
        Function to run data class objects on data to prepare them for the model 
        
        Parameters
        ----------
        X : pd.DataFrame
            feature data to vectorize
        y : pd.Series
            target column. Only needed if ``train_on=True`` and pipeline contains :class:`Selector` or :class:`Sampler`. Otherwise, just input ``None``
        train_on : bool
            the data will always be transformed. If ``train_on=True``, the transformers will be fit_transformed

        Returns
        -------
        X : pd.DataFrame
            transformed feature data
        y : pd.Series
            transformed target column. Only differes from input if ``train_on=True`` and pipeline contains :class:`Sampler`
        """
        if self.vectorizer is not None:
            X = self._auto_vectorizing(X, train_on=train_on)
        if self.scaler is not None:
            X = self.scaler.scale(X, train_on=train_on)
        if self.selector is not None:
            X = self.selector.select(X, y, train_on=train_on)
        if self.sampler is not None and train_on:
            X, y = self.sampler.sample(X, y)
        self._data_classes_trained = True
        return X, y

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        x_train_pre, y_train_pre = self.data_prepare(x_train, y_train, train_on=True)
        self._feature_names = list(x_train_pre.columns)
        return super().fit(x_train_pre, y_train_pre, **kwargs)
    
    def fit_warm_start(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        x_train_pre, y_train_pre = self.data_prepare(x_train, y_train, train_on = not self._data_classes_trained)
        self._feature_names = list(x_train_pre.columns)
        return super().fit(x_train_pre, y_train_pre, **kwargs)

    def predict(self, x_test: pd.DataFrame) -> list:
        x_test_pre, _ = self.data_prepare(x_test, None, train_on=False)
        return super().predict(x_test_pre)

    def predict_proba(self, x_test: pd.DataFrame) -> list:
        x_test_pre, _ = self.data_prepare(x_test, None, train_on=False)
        return super().predict_proba(x_test_pre)

    def get_params(self, deep: bool = True) -> dict[str, any]:
        return dict(self.steps)


# class factory 
def create_pipeline(model: Classifier | Regressor,  vectorizer: str | Embeddings_builder | None = None, scaler: str | Scaler | None = None, selector: str | tuple[str, int] | Selector | None = None, sampler: str | Sampler | SamplerPipeline | None = None, model_name: str = "pipe"):
    """
    Parameters
    ----------
    model : Classifier or Regressor class object
        Model used in pipeline (:class:`Classifier` or :class:`Regressor`)
    vectorizer : str, Embeddings_builder, or None
        object or algorithm of :class:`Embeddings_builder` class which will be used for automatic string column vectorizing (None for no vectorizing)
    scaler : str, Scaler, or None
        object or algorithm of :class:`Scaler` class for scaling the data (None for no scaling)
    selector : str, Selector, or None
        object or algorithm of :class:`Selector` class for feature selection (None for no selecting)
    sampler : str, Sampler, or None
        object or algorithm of :class:`Sampler` class for sampling the train data (None for no sampling)
    model_name : str
        name of the model
    """
    if not issubclass(model.__class__, (Classifier, Regressor)):
        raise ValueError(f"wrong input '{model}' (type: {type(Model)}) for model")

    class DynamicPipeline(BasePipeline, model.__class__.__base__):
        pass

    # quick solution: discrete vs continuous values
    if type(model).__base__ == Regressor:
        sampler = None

    return DynamicPipeline(model=model, vectorizer=vectorizer, scaler=scaler, selector=selector, sampler=sampler, model_name=model_name)
