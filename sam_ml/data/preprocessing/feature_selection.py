import os
import sys
import warnings
from typing import Literal

import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    chi2,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sam_ml.config import setup_logger

from ..main_data import Data

logger = setup_logger(__name__)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affects subprocesses


class Selector(Data):
    """ feature selection algorithm Wrapper class """

    def __init__(self, algorithm: Literal["kbest", "kbest_chi2", "pca", "wrapper", "sequential", "select_model", "rfe", "rfecv"] = "kbest", num_features: int = 10, estimator = LinearSVC(penalty="l1", dual=False), **kwargs):
        """
        Parameters
        ----------
        algorithm : {"kbest", "kbest_chi2", "pca", "wrapper", "sequential", "select_model", "rfe", "rfecv"}, \
                default="kbest"
            which selecting algorithm to use:
            - 'kbest': SelectKBest
            - 'kbest_chi2': SelectKBest with score_func=chi2 (only non-negative values)
            - 'pca': PCA (new column names after transformation)
            - 'wrapper': uses p-values of Ordinary Linear Model from statsmodels library (no num_features parameter -> problems with too many features)
            - 'sequential': SequentialFeatureSelector
            - 'select_model': SelectFromModel (meta-transformer for selecting features based on importance weights)
            - 'rfe': RFE (recursive feature elimination)
            - 'rfecv': RFECV (recursive feature elimination with cross-validation)

        num_features : int, \
                default=10
            number of features to select
        estimator : estimator instance
            parameter is needed for SequentialFeatureSelector, SelectFromModel, RFE, RFECV (default: LinearSVC)
        \*\*kwargs:
            additional parameters for selector
        """
        self._num_features = num_features
        self._selected_features = []

        if algorithm == "kbest":
            selector = SelectKBest(k=num_features, **kwargs)
        elif algorithm == "kbest_chi2":
            selector = SelectKBest(k=num_features, score_func=chi2, **kwargs)
        elif algorithm == "pca":
            selector = PCA(n_components=num_features, random_state=42, **kwargs)
        elif algorithm == "wrapper":
            selector = {"pvalue_limit": 0.5}
        elif algorithm == "sequential":
            selector = SequentialFeatureSelector(estimator, n_features_to_select=num_features, **kwargs)
        elif algorithm == "select_model":
            selector = SelectFromModel(estimator, max_features=num_features, **kwargs)
        elif algorithm == "rfe":
            selector = RFE(estimator, n_features_to_select=num_features, **kwargs)
        elif algorithm == "rfecv":
            selector = RFECV(estimator, min_features_to_select=num_features, **kwargs)
        else:
            raise ValueError(f"algorithm='{algorithm}' is not supported")
        
        super().__init__(algorithm, selector)

    @property
    def num_features(self) -> int:
        """
        Returns
        -------
        num_features : int
            number of features to select
        """
        return self._num_features
    
    @property
    def selected_features(self) -> list[str]:
        """
        Returns
        -------
        selected_features : list[str]
            list with selected feature names
        """
        return self._selected_features

    @staticmethod
    def params() -> dict:
        """
        Function to get the possible/recommended parameter values for the class

        Returns
        -------
        param : dict
            possible values for the parameter "algorithm" and recommended values for "estimator"

        Examples
        --------
        >>> # get possible/recommended parameters
        >>> from sam_ml.data.preprocessing import Selector
        >>>
        >>> # first way without class object
        >>> params1 = Selector.params()
        >>> print(params1)
        {"algorithm": ["kbest", ...], "estimator": [LinearSVC(penalty="l1", dual=False), ...]}
        >>> # second way with class object
        >>> model = Selector()
        >>> params2 = model.params()
        >>> print(params2)
        {"algorithm": ["kbest", ...], "estimator": [LinearSVC(penalty="l1", dual=False), ...]}
        """
        param = {
            "algorithm": ["kbest", "kbest_chi2", "pca", "wrapper", "sequential", "select_model", "rfe", "rfecv"], 
            "estimator": [LinearSVC(penalty="l1", dual=False), LogisticRegression(), ExtraTreesClassifier(n_estimators=50)]
        }
        return param

    def get_params(self, deep: bool = True) -> dict:
        class_params = {"algorithm": self.algorithm, "num_features": self.num_features}
        if self.algorithm == "wrapper":
            return class_params | self.transformer
        else:
            selector_params = self.transformer.get_params(deep)
            if self.algorithm in ("kbest", "kbest_chi2"):
                selector_params.pop("k")
            elif self.algorithm in ("pca"):
                selector_params.pop("n_components")
            elif self.algorithm in ("sequential", "rfe"):
                selector_params.pop("n_features_to_select")
            elif self.algorithm in ("select_model"):
                selector_params.pop("max_features")
            elif self.algorithm in ("rfecv"):
                selector_params.pop("min_features_to_select")

            return class_params | selector_params

    def set_params(self, **params):
        if self.algorithm == "wrapper":
            self._transformer = params
        else:
            self._transformer.set_params(**params)
        return self
    
    def select(self, X: pd.DataFrame, y: pd.DataFrame = None, train_on: bool = True) -> pd.DataFrame:
        """
        Select the best features from data

        Parameters
        ----------
        X : pd.DataFrame
            X data to use for feature selection
        y : pd.Series, \
                default=None
            y data to use for feature selection. Only needed when ``train_on=True``
        train_on : bool, \
                default=True
            If ``True``, the estimator instance will be trained to select the best features for the given y. Otherwise, it just selects the correct columns from X.

        Returns
        -------
        X_selected : pd.DataFrame
            X with only the selected columns

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>>
        >>> # select features
        >>> from sam_ml.data.preprocessing import Selector
        >>>
        >>> model = Selector(num_features=2)
        >>> x_train_selected = model.select(x_train, y_train) # train selector
        >>> x_test_selected = model.select(x_test, train_on=False) # select test data
        >>> print("all feature names:")
        >>> print(list(x_train.columns))
        >>> print()
        >>> print("selected features:")
        >>> print(list(x_train_selected.columns))
        all feature names:
        ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        <BLANKLINE>
        selected features:
        ['petal length (cm)', 'petal width (cm)']
        """
        if len(X.columns) < self.num_features:
            logger.warning("the number of features that shall be selected is greater than the number of features in X --> return X")
            self._selected_features = X.columns
            return X

        logger.debug("selecting features - started")
        if train_on:
            if self.algorithm == "wrapper":
                self._selected_features = self.__wrapper_select(X, y, **self._transformer)
            else:
                self._transformer.fit(X.values, y)
                self._selected_features = self._transformer.get_feature_names_out(X.columns)
        
        if self.algorithm == "wrapper":
            X_selected = X[self.selected_features]
        else:
            X_selected = pd.DataFrame(self._transformer.transform(X), columns=self.selected_features)

        logger.debug("selecting features - finished")
        return X_selected

    def __wrapper_select(self, X: pd.DataFrame, y: pd.DataFrame, pvalue_limit: float = 0.5, **kwargs) -> list[str]:
        """
        Select the best feature names from data

        uses p-values of Ordinary Linear Model from statsmodels library to iteratively eliminate features

        Parameters
        ----------
        X : pd.DataFrame
            X data to use for feature selection
        y : pd.Series
            y data to use for feature selection
        pvalue_limit : float, \
                default=0.5
            if p value of feature is above the pvalue_limit, it will be removed

        Returns
        -------
        selected_features : list[str]
            list with selected feature names
        """
        selected_features = list(X.columns)
        y = list(y)
        pmax = 1
        while selected_features:
            p= []
            X_new = X[selected_features]
            X_new = sm.add_constant(X_new)
            model = sm.OLS(y,X_new).fit()
            p = pd.Series(model.pvalues.values[1:],index = selected_features)      
            pmax = max(p)
            feature_pmax = p.idxmax()
            if(pmax>pvalue_limit):
                selected_features.remove(feature_pmax)
            else:
                break
        if len(selected_features) == len(X.columns):
            logger.warning("the wrapper algorithm selected all features")
        return selected_features
