import copy

import pandas as pd

from sam_ml.config import setup_logger
from sam_ml.data.main_data import Data
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

    def __init__(self, model: Classifier | Regressor,  vectorizer: str | Embeddings_builder | None, scaler: str | Scaler | None, selector: str | tuple[str, int] | Selector | None, sampler: str | Sampler | SamplerPipeline | None, model_name: str):
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
            object, tuple of algorithm and feature number, or algorithm of :class:`Selector` class for feature selection (None for no selecting)
        sampler : str, Sampler, SamplerPipeline, or None
            object or algorithm of :class:`Sampler` / :class:`SamplerPipeline` class for sampling the train data (None for no sampling)
        model_name : str
            name of the model
        """
        super().__init__(model_object=model.model, model_name=model_name, model_type=model.model_type, grid=model.grid)

        self._inherit_from_model(model)

        self.__model = model

        self._vectorizer = self._validate_component(vectorizer, Embeddings_builder)
        self._scaler = self._validate_component(scaler, Scaler)
        self._selector = self._validate_component(selector, Selector)
        self._sampler = self._validate_component(sampler, Sampler, SamplerPipeline)

        self._vectorizer_dict: dict[str, Embeddings_builder] = {}

        # keep track if model was trained for warm_start
        self._data_classes_trained: bool = False
        # list of detected string columns
        self._string_columns: list[str] = []

    def __repr__(self) -> str:
        params: str = ""
        for step in self.steps:
            params += step[0]+"="+step[1].__str__()+", "

        params += f"model_name='{self.model_name}'"

        return f"Pipeline({params})"
    
    def _inherit_from_model(self, model: Classifier | Regressor):
        """ 
        Function to inherit methods and attributes from model
        
        Parameters
        ----------
        model : Classifier or Regressor class object
            model used in pipeline (:class:`Classifier` or :class:`Regressor`)
        """
        for attribute_name in dir(model):
            attribute_value = getattr(model, attribute_name)

            # Check if the attribute is a method or a variable (excluding private attributes)
            if callable(attribute_value) and not attribute_name.startswith("__"):
                if not hasattr(self, attribute_name):
                    setattr(self, attribute_name, attribute_value)
            elif not attribute_name.startswith("__"):
                if not hasattr(self, attribute_name):
                    self.__dict__[attribute_name] = attribute_value
    
    @staticmethod
    def _validate_component(component: str | tuple | Data, component_class: Data, pipeline_class: Data = None) -> Data:
        """
        Function to create the data preprocessing steps

        Parameters
        ----------
        component : str, tuple, or Data object
            \_\_init\_\_-method input value for data preprocessing component
        component_class : Data object
            class of component like `Sampler`, `Selector`, ...
        pipeline_class : Data object, default=None
            currently, only used for SamplerPipeline class input in \_\_init\_\_-method

        Returns
        -------
        Data object
        """
        if isinstance(component, str):
            if component in component_class.params()["algorithm"]:
                return component_class(algorithm=component)
            
            # special case for sampler pipeline
            elif not pipeline_class is None:
                return pipeline_class(algorithm=component)

        elif isinstance(component, component_class) or component is None:
            return component
        
        # special case for selector
        elif isinstance(component, tuple) and len(component) == 2:
            if component[0] in component_class.params()["algorithm"] and type(component[1])==int:
                if component[1] > 0:
                    return component_class(algorithm=component[0], num_features=component[1])
                else:
                    raise ValueError(f"wrong input '{component}' for {component_class.__name__} -> integer in tuple has to be greater 0")
                
        raise ValueError(f"wrong input '{component}' for {component_class.__name__}")
    
    @property
    def string_columns(self) -> list[str]:
        """
        Returns
        -------
        string_columns : list[str]
            list with detected string columns that are used in auto-vectorizing
        """
        return self._string_columns
    
    @property
    def data_classes_trained(self) -> bool:
        """
        Returns
        -------
        data_classes_trained : bool
            If ``True``, the preprocessing step classes are fitted. Important for methods that use warm_start
        """
        return self._data_classes_trained

    @property
    def steps(self) -> list[tuple[str, any]]:
        """
        Returns
        -------
        steps : list[tuple[str, any]]
            list with preprocessing + model pipeline steps as tuples
        """
        return [("vectorizer", self._vectorizer), ("scaler", self._scaler), ("selector", self._selector), ("sampler", self._sampler), ("model", self.__model)]
    
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
            self._vectorizer_dict = dict(zip(self._string_columns, [copy.deepcopy(self._vectorizer) for i in range(len(string_columns))]))

        for col in self._string_columns:
            X = pd.concat([X, self._vectorizer_dict[col].vectorize(X[col], train_on=train_on)], axis=1)
        X_vectorized = X.drop(columns=self._string_columns)

        return X_vectorized

    def _data_prepare(self, X: pd.DataFrame, y: pd.Series, train_on: bool = True) -> tuple[pd.DataFrame, pd.Series]:
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
        if self._vectorizer is not None:
            X = self._auto_vectorizing(X, train_on=train_on)
        if self._scaler is not None:
            X = self._scaler.scale(X, train_on=train_on)
        if self._selector is not None:
            X = self._selector.select(X, y, train_on=train_on)
        if self._sampler is not None and train_on:
            X, y = self._sampler.sample(X, y)
        self._data_classes_trained = True
        return X, y

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        x_train_pre, y_train_pre = self._data_prepare(x_train, y_train, train_on=True)
        self._feature_names = list(x_train_pre.columns)
        return super().fit(x_train_pre, y_train_pre, **kwargs)
    
    def fit_warm_start(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        x_train_pre, y_train_pre = self._data_prepare(x_train, y_train, train_on = not self._data_classes_trained)
        self._feature_names = list(x_train_pre.columns)
        return super().fit(x_train_pre, y_train_pre, **kwargs)

    def predict(self, x_test: pd.DataFrame) -> list:
        x_test_pre, _ = self._data_prepare(x_test, None, train_on=False)
        return super().predict(x_test_pre)

    def predict_proba(self, x_test: pd.DataFrame) -> list:
        x_test_pre, _ = self._data_prepare(x_test, None, train_on=False)
        return super().predict_proba(x_test_pre)

    def get_params(self, deep: bool = True) -> dict[str, any]:
        return dict(self.steps) | {"model_name": self.model_name}


# class factory 
def create_pipeline(model: Classifier | Regressor,  vectorizer: str | Embeddings_builder | None = None, scaler: str | Scaler | None = None, selector: str | tuple[str, int] | Selector | None = None, sampler: str | Sampler | SamplerPipeline | None = None, model_name: str = "pipe") -> BasePipeline:
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
    sampler : str, Sampler, SamplerPipeline, or None
        object or algorithm of :class:`Sampler` / :class:`SamplerPipeline` class for sampling the train data (None for no sampling). For Regressor model, always ``None`` (will be implemented in the future).
    model_name : str
        name of the model

    Returns
    -------
    DynamicPipeline object which inherits from the model parent class and BasePipeline

    Examples
    --------
    >>> # load data (replace with own data)
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=3000, n_features=4, n_classes=2, weights=[0.9], random_state=42)
    >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(y)
    >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
    >>> 
    >>> # train and evaluate model pipeline
    >>> from sam_ml.models import create_pipeline
    >>> from sam_ml.models.classifier import LR
    >>>
    >>> model = create_pipeline(LR(), scaler="standard", sampler="SMOTE_rus_20_50")
    >>> model.train(x_train, y_train)
    >>> scores = model.evaluate(x_test, y_test)
    Train score: 0.9625 - Train time: 0:00:00
    accuracy: 0.9583333333333334
    precision: 0.8563762626262625
    recall: 0.9377241446156828
    s_score: 0.9603691957893064
    l_score: 0.9989822522866367
    <BLANKLINE>
    classification report: 
                    precision   recall  f1-score    support
    <BLANKLINE>
            0       0.99        0.96    0.98        543
            1       0.72        0.91    0.81        57
    <BLANKLINE>
    accuracy                            0.96        600
    macro avg       0.86        0.94    0.89        600
    weighted avg    0.97        0.96    0.96        600
    <BLANKLINE>
    """
    if not issubclass(model.__class__, (Classifier, Regressor)):
        raise ValueError(f"wrong input '{model}' (type: {type(Model)}) for model")

    class DynamicPipeline(BasePipeline, model.__class__.__base__):
        pass

    # quick solution: discrete vs continuous values
    if type(model).__base__ == Regressor:
        sampler = None

    return DynamicPipeline(model=model, vectorizer=vectorizer, scaler=scaler, selector=selector, sampler=sampler, model_name=model_name)
