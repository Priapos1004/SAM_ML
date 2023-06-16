import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel,
                                       SelectKBest, SequentialFeatureSelector,
                                       chi2)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sam_ml.config import setup_logger

logger = setup_logger(__name__)


class Selector:
    """ feature selection algorithm Wrapper class """

    def __init__(self, algorithm: str = "kbest", num_features: int = 10, estimator = LinearSVC(penalty="l1", dual=False), **kwargs):
        """
        @params:
            algorithm:
                'kbest': SelectKBest
                'kbest_chi2': SelectKBest with score_func=chi2 (only non-negative values)
                'pca': PCA (new column names after transformation)
                'wrapper': uses p-values of Ordinary Linear Model from statsmodels library (no num_features parameter -> problems with too many features)
                'sequential': SequentialFeatureSelector
                'select_model': SelectFromModel (meta-transformer for selecting features based on importance weights)
                'rfe': RFE (recursive feature elimination)
                'rfecv': RFECV (recursive feature elimination with cross-validation)
            
            estimator:
                parameter is needed for SequentialFeatureSelector, SelectFromModel, RFE, RFECV (default: LinearSVC)
            
            **kwargs:
                additional parameters for selector
        """
        self.algorithm = algorithm
        self.num_features = num_features
        self._grid: dict[str, list] = {} # for pipeline structure

        if algorithm == "kbest":
            self.selector = SelectKBest(k=num_features, **kwargs)
        elif algorithm == "kbest_chi2":
            self.selector = SelectKBest(k=num_features, score_func=chi2, **kwargs)
        elif algorithm == "pca":
            self.selector = PCA(n_components=num_features, random_state=42, **kwargs)
        elif algorithm == "wrapper":
            self.selector = {"pvalue_limit": 0.5}
        elif algorithm == "sequential":
            self.selector = SequentialFeatureSelector(estimator, n_features_to_select=num_features, **kwargs)
        elif algorithm == "select_model":
            self.selector = SelectFromModel(estimator, max_features=num_features, **kwargs)
        elif algorithm == "rfe":
            self.selector = RFE(estimator, n_features_to_select=num_features, **kwargs)
        elif algorithm == "rfecv":
            self.selector = RFECV(estimator, min_features_to_select=num_features, **kwargs)
        else:
            logger.error(f"algorithm='{algorithm}' does not exist -> using SelectKBest algorithm instead")
            self.selector = SelectKBest(k=num_features)
            self.algorithm = "kbest"

    def __repr__(self) -> str:
        selector_params: str = ""
        param_dict = self.get_params(False)
        for key in param_dict:
            if type(param_dict[key]) == str:
                selector_params += key+"='"+str(param_dict[key])+"', "
            else:
                selector_params += key+"="+str(param_dict[key])+", "
        return f"Selector({selector_params})"

    @staticmethod
    def params() -> dict:
        """
        @return:
            possible/recommended values for the parameters
        """
        param = {
            "algorithm": ["kbest", "kbest_chi2", "pca", "wrapper", "sequential", "select_model", "rfe", "rfecv"], 
            "estimator": [LinearSVC(penalty="l1", dual=False), LogisticRegression(), ExtraTreesClassifier(n_estimators=50)]
        }
        return param

    def get_params(self, deep: bool = True):
        class_params = {"algorithm": self.algorithm, "num_features": self.num_features}
        if self.algorithm == "wrapper":
            return class_params | self.selector
        else:
            selector_params = self.selector.get_params(deep)
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
            self.selector = params
        else:
            self.selector.set_params(**params)
        return self
    
    def select(self, X: pd.DataFrame, y: pd.DataFrame = None, train_on: bool = True) -> pd.DataFrame:
        """
        for training: the y data is also needed
        """
        if len(X.columns) < self.num_features:
            logger.warning("the number of features that shall be selected is greater than the number of features in X --> return X")
            self.selected_features = X.columns
            return X

        logger.debug("selecting features - started")
        if train_on:
            if self.algorithm == "wrapper":
                self.selected_features = self.__wrapper_select(X, y, **self.selector)
            else:
                self.selector.fit(X.values, y)
                self.selected_features = self.selector.get_feature_names_out(X.columns)
        
        if self.algorithm == "wrapper":
            X_selected = X[self.selected_features]
        else:
            X_selected = pd.DataFrame(self.selector.transform(X), columns=self.selected_features)

        logger.debug("selecting features - finished")
        return X_selected

    def __wrapper_select(self, X: pd.DataFrame, y: pd.DataFrame, pvalue_limit: float = 0.5, **kwargs) -> list:
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
