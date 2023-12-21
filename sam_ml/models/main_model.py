import inspect
import pickle
import time
from abc import abstractmethod
from copy import deepcopy
from datetime import timedelta
from statistics import mean
from typing import Callable

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from matplotlib import pyplot as plt
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_validate
from tqdm import tqdm

from sam_ml.config import get_n_jobs, setup_logger

SMAC_INSTALLED: bool
try:
    from smac import HyperparameterOptimizationFacade, Scenario
    SMAC_INSTALLED = True
except:
    SMAC_INSTALLED = False

logger = setup_logger(__name__)

class Model:
    """ Model parent class {abstract} """

    def __init__(self, model_object, model_name: str, model_type: str, grid: ConfigurationSpace):
        """
        Parameters
        ----------
        model_object : model object
            model with 'fit', 'predict', 'set_params', and 'get_params' method (see sklearn API)
        model_name : str
            name of the model
        model_type : str
            kind of estimator (e.g. 'RFC' for RandomForestClassifier)
        grid : ConfigurationSpace
            hyperparameter grid for the model
        """
        self._model = model_object
        self._model_name = model_name
        self._model_type = model_type
        self._grid = grid
        self._train_score: float = None
        self._train_time: str = None
        self._feature_names: list[str] = []
        self._cv_scores: dict[str, float] = {}
        self._rCVsearch_results: pd.DataFrame|None = None

    def __repr__(self) -> str:
        params: str = ""
        param_dict = self._changed_parameters()
        for key in param_dict:
            if type(param_dict[key]) == str:
                params+= key+"='"+str(param_dict[key])+"', "
            else:
                params+= key+"="+str(param_dict[key])+", "
        params += f"model_name='{self.model_name}'"

        return f"{self.model_type}({params})"
    
    def _changed_parameters(self):
        """
        Function to get parameters that differ from the default ones

        Returns
        -------
        dictionary of model parameter that are different from default values
        """
        params = self.get_params(deep=False)
        init_params = inspect.signature(self.__init__).parameters
        init_params = {name: param.default for name, param in init_params.items()}

        init_params_estimator = inspect.signature(self.model.__init__).parameters
        init_params_estimator = {name: param.default for name, param in init_params_estimator.items()}

        def has_changed(k, v):
            if k not in init_params:  # happens if k is part of a **kwargs
                if k not in init_params_estimator: # happens if k is part of a **kwargs
                    return True
                else:
                    if v != init_params_estimator[k]:
                        return True
                    else:
                        return False

            if init_params[k] == inspect._empty:  # k has no default value
                return True
            elif init_params[k] != v:
                return True
            
            return False

        return {k: v for k, v in params.items() if has_changed(k, v)}
    
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
            name of the model. Used in loading bars and dictionaries as identifier of the model
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
    def grid(self) -> ConfigurationSpace:
        """
        Returns
        -------
        grid : ConfigurationSpace
            hyperparameter tuning grid of the model
        """
        return self._grid
    
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
    
    @property
    def cv_scores(self) -> dict[str, float]:
        """
        Returns
        -------
        cv_scores : dict[str, float]
            dictionary with cross validation results
        """
        return self._cv_scores
    
    @property
    def rCVsearch_results(self) -> pd.DataFrame|None:
        """
        Returns
        -------
        rCVsearch_results : pd.DataFrame or None
            results from randomCV hyperparameter tuning. Is ``None`` if randomCVsearch was not used yet.
        """
        return self._rCVsearch_results
    
    def replace_grid(self, new_grid: ConfigurationSpace):
        """
        Function to replace self.grid
        
        See `ConfigurationSpace documentation <https://automl.github.io/ConfigSpace/main/>`_.

        Parameters
        ----------
        new_grid : ConfigurationSpace
            new grid to replace the old one with

        Returns
        -------
        changes self.grid variable

        Examples
        --------
        >>> from ConfigSpace import ConfigurationSpace, Categorical, Float
        >>> from sam_ml.models.classifier import LDA
        >>>
        >>> model = LDA()
        >>> new_grid = ConfigurationSpace(
        ...     seed=42,
        ...     space={
        ...         "solver": Categorical("solver", ["lsqr", "eigen"]),
        ...         "shrinkage": Float("shrinkage", (0, 0.5)),
        ...     })
        >>> model.replace_grid(new_grid)
        """
        self._grid = new_grid
    
    def get_random_config(self) -> dict:
        """
        Function to generate one grid configuration

        Returns
        -------
        config : dict
            dictionary of random parameter configuration from grid

        Examples
        --------
        >>> from sam_ml.models.classifier import LR
        >>> 
        >>> model = LR()
        >>> model.get_random_config()
        {'C': 0.31489116479568624,
        'penalty': 'elasticnet',
        'solver': 'saga',
        'l1_ratio': 0.6026718993550663}
        """
        return dict(self.grid.sample_configuration(1))
    
    def get_random_configs(self, n_trails: int) -> list[dict]:
        """
        Function to generate grid configurations

        Parameters
        ----------
        n_trails : int
            number of grid configurations

        Returns
        -------
        configs : list
            list with sets of random parameter from grid

        Notes
        -----
        filter out duplicates -> could be less than n_trails

        Examples
        --------
        >>> from sam_ml.models.classifier import LR
        >>> 
        >>> model = LR()
        >>> model.get_random_configs(3)
        [Configuration(values={
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
        }),
        Configuration(values={
            'C': 2.5378155082656657,
            'penalty': 'l2',
            'solver': 'saga',
        }),
        Configuration(values={
            'C': 2.801635158716261,
            'penalty': 'l2',
            'solver': 'lbfgs',
        })]
        """
        if n_trails<1:
            raise ValueError(f"n_trails has to be greater 0, but {n_trails}<1")
        
        configs = [self._grid.get_default_configuration()]
        if n_trails == 2:
            configs += [self._grid.sample_configuration(n_trails-1)]
        else:
            configs += self._grid.sample_configuration(n_trails-1)
        # remove duplicates
        configs = list(dict.fromkeys(configs))
        return configs
    
    def _print_scores(self, scores: dict, y_test: pd.Series, pred: list):
        """
        Function to print out the values of a dictionary

        Parameters
        ----------
        scores: dict
            dictionary with score names and values
        y_test, pred : pd.Series, list
            Data to evaluate model

        Returns
        -------
        key-value pairs in console, format: 
        
        key1: value1

        key2: value2

        ...
        """
        for key in scores:
            print(f"{key}: {scores[key]}")
    
    @abstractmethod
    def _make_scorer(
        self,
        custom_score: Callable | None,
        **kwargs,
    ) -> dict[str, Callable]:
        """
        Function to create a dictionary with scorer for the crossvalidation
        
        Parameters
        ----------
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.
        \*\*kwargs:
            additional parameters from child-class

        Returns
        -------
        scorer : dict[str, Callable]
            dictionary with scorer functions
        """
        pass

    @abstractmethod
    def _make_cv_scores(
            self,
            score: dict,
            custom_score: Callable | None,
    ) -> dict[str, float]:
        """
        Function to create from the crossvalidation results a dictionary
        
        Parameters
        ----------
        score : dict
            crossvalidation average column results
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        cv_scores : dict
            restructured dictionary
        """
        pass

    @abstractmethod
    def _get_score(
        self,
        scoring: str,
        y_test: pd.Series,
        pred: list,
        **kwargs,
    ) -> float:
        """ 
        Calculate a score for given y true and y prediction values

        Parameters
        ----------
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`
        y_test, pred : pd.Series, pd.Series
            Data to evaluate model

        Returns
        -------
        score : float 
            metrics score value
        """
        pass
    
    @abstractmethod
    def _get_all_scores(
        self,
        y_test: pd.Series,
        pred: list,
        custom_score: Callable,
        **kwargs,
    ) -> dict[str, float]:
        """
        Function to create multiple scores for given y_true-y_pred pairs

        Parameters
        ----------
        y_test, pred : pd.Series, list
            Data to evaluate model
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.
        \*\*kwargs:
            additional parameters from child-class

        Returns
        -------
        scores : dict 
            dictionary with score names as keys and score values as values
        """
        pass

    def train(self, x_train: pd.DataFrame, y_train: pd.Series, console_out: bool = True, **kwargs) -> tuple[float, str]:
        """
        Function to train the model

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train model
        console_out : bool, default=True
            shall the score and time be printed out
        \*\*kwargs:
            additional parameters from child-class for ``evaluate_score`` method

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
        console_out : bool, default=True
            shall the score and time be printed out
        \*\*kwargs:
            additional parameters from child-class for ``evaluate_score`` method

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
        \*\*kwargs:
            additional parameters from child-class for ``fit`` method

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
        \*\*kwargs:
            additional parameters from child-class for ``fit`` method

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
        deep : bool, default=True
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
        \*\*params : dict
            Estimator parameters

        Returns
        -------
        self : estimator instance
            Estimator instance
        """
        self.model.set_params(**params)
        return self
    
    def evaluate(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        console_out: bool,
        custom_score: Callable,
        **kwargs,
    ) -> dict[str, float]:
        """
        Function to create multiple scores with predict function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate model
        console_out : bool
            shall the result of the different scores and a classification_report be printed into the console
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.
        \*\*kwargs:
            additional parameters from child-class for ``_get_all_scores`` method

        Returns
        -------
        scores : dict 
            dictionary of the format from the self._get_all_scores function
        """
        pred = self.predict(x_test)
        scores = self._get_all_scores(y_test, pred, custom_score=custom_score, **kwargs)

        if console_out:
            self._print_scores(scores, y_test=y_test, pred=pred)

        return scores

    def evaluate_score(
        self,
        scoring: str | Callable,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs,
    ) -> float:
        """
        Function to create a score with self.__get_score of model

        Parameters
        ----------
        scoring : str or callable (custom score)
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`
        x_test, y_test : pd.DataFrame, pd.Series
            Data for evaluating the model
        \*\*kwargs:
            additional parameters from child-class for ``_get_score`` method

        Returns
        -------
        score : float
            metrics score value
        """
        pred = self.predict(x_test)
        score = self._get_score(scoring, y_test, pred, **kwargs)

        return score
    
    def cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int,
        console_out: bool,
        custom_score: Callable | None,
        **kwargs,
    ) -> dict[str, float]:
        """
        Random split crossvalidation

        Parameters
        ----------
        X, y : pd.DataFrame, pd.Series
            Data to cross validate on
        cv_num : int
            number of different random splits
        console_out : bool
            shall the result dataframe of the different scores for the different runs be printed
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.
        \*\*kwargs:
            additional parameters from child-class for ``make_scorer`` method

        Returns
        -------
        scores : dict 
            dictionary of the format from the self._make_cv_scores function

        The scores are also saved in ``self.cv_scores``.
        """
        logger.debug(f"cross validation {self.model_name} - started")

        scorer = self._make_scorer(**kwargs, custom_score=custom_score)

        cv_scores = cross_validate(
            self,
            X,
            y,
            scoring=scorer,
            cv=cv_num,
            return_train_score=True,
            n_jobs=get_n_jobs(),
        )

        pd_scores = pd.DataFrame(cv_scores).transpose()
        pd_scores["average"] = pd_scores.mean(numeric_only=True, axis=1)

        score = pd_scores["average"]

        self._cv_scores = self._make_cv_scores(score, custom_score)

        logger.debug(f"cross validation {self.model_name} - finished")

        if console_out:
            print()
            print(pd_scores)

        return self._cv_scores
    
    def cross_validation_small_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        leave_loadbar: bool,
        console_out: bool,
        custom_score: Callable | None,
        **kwargs,
    ) -> dict[str, float]:
        """
        One-vs-all cross validation for small datasets

        In the cross_validation_small_data-method, the model will be trained on all datapoints except one and then tested on this last one. 
        This will be repeated for all datapoints so that we have our predictions for all datapoints.

        Advantage: optimal use of information for training

        Disadvantage: long train time

        This concept is very useful for small datasets (recommended: datapoints < 150) because the long train time is still not too long and 
        especially with a small amount of information for the model, it is important to use all the information one has for the training.

        Parameters
        ----------
        X, y : pd.DataFrame, pd.Series
            Data to cross validate on
        leave_loadbar : bool
            shall the loading bar of the training be visible after training (True - load bar will still be visible)
        console_out : bool
            shall the result of the different scores and a classification_report be printed into the console
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.
        \*\*kwargs:
            additional parameters from child-class for ``_get_all_scores`` method

        Returns
        -------
        scores : dict 
            dictionary of the format from the self._get_all_scores function

        The scores are also saved in ``self.cv_scores``.
        """
        logger.debug(f"cross validation {self.model_name} - started")

        predictions = []
        t_scores = []
        t_times = []
        
        for idx in tqdm(X.index, desc=self.model_name, leave=leave_loadbar):
            x_train = X.drop(idx)
            y_train = y.drop(idx)
            x_test = X.loc[[idx]]

            train_score, train_time = self.train(x_train, y_train, console_out=False)
            prediction = self.predict(x_test)

            predictions += prediction
            t_scores.append(train_score)
            t_times.append(train_time)

        self._cv_scores = self._get_all_scores(y_test=y, pred=predictions, **kwargs, custom_score=custom_score)
        avg_train_score = mean(t_scores)
        avg_train_time = str(timedelta(seconds=round(sum(map(lambda f: int(f[0])*3600 + int(f[1])*60 + int(f[2]), map(lambda f: f.split(':'), t_times)))/len(t_times))))

        self._cv_scores.update({
            "train_score": avg_train_score,
            "train_time": avg_train_time,
        })

        if console_out:
            self._print_scores(self._cv_scores, y_test=y, pred=predictions)

        logger.debug(f"cross validation {self.model_name} - finished")

        return self._cv_scores
    
    def feature_importance(self) -> plt.show:
        """
        Function to generate a matplotlib plot of the top45 feature importance from the model. 
        You can only use the method if you trained your model before.

        Returns
        -------
        plt.show object

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>> 
        >>> # train and plot features of model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> model.train(X, y)
        >>> model.feature_importance()
        """
        if not self.feature_names:
            raise NotFittedError("You have to first train the classifier before getting the feature importance (with train-method)")

        if self.model_type == "MLPC":
            importances = [np.mean(i) for i in self.model.coefs_[0]]  # MLP Classifier
        elif self.model_type in ("DTC", "RFC", "GBM", "CBC", "ABC", "ETC", "XGBC")+("RFR", "DTR", "ETR", "XGBR"):
            importances = self.model.feature_importances_
        elif self.model_type in ("KNC", "GNB", "BNB", "GPC", "QDA", "BC"):
            logger.warning(f"{self.model_type} does not have a feature importance")
            return
        else:
            importances = self.model.coef_[0]  # "normal"

        # top45 features
        feature_importances = pd.Series(importances, index=self.feature_names).sort_values(ascending=False).head(45)

        fig, ax = plt.subplots()
        if self.model_type in ("RFC", "GBM", "ETC")+("RFR", "ETR"):
            if self.model_type in ("RFC", "ETC")+("RFR", "ETR"):
                std = np.std(
                    [tree.feature_importances_ for tree in self.model.estimators_], axis=0,
                )
            elif self.model_type == "GBM":
                std = np.std(
                    [tree[0].feature_importances_ for tree in self.model.estimators_], axis=0,
                )
            feature_importances.plot.bar(yerr=std, ax=ax)
        else:
            feature_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances of " + str(self.model_name))
        ax.set_ylabel("use of coefficients as importance scores")
        fig.tight_layout()
        plt.show()

    def smac_search(
        self,
        x_train: pd.DataFrame, 
        y_train: pd.Series,
        scoring: str | Callable,
        n_trails: int,
        cv_num: int,
        small_data_eval: bool,
        walltime_limit: int,
        log_level: int,
        **kwargs,
    ) -> Configuration:
        """
        Hyperparametertuning with SMAC library HyperparameterOptimizationFacade [can only be used in the sam_ml version with swig]

        The smac_search-method will more "intelligent" search your hyperparameter space than the randomCVsearch and 
        returns the best hyperparameter set. Additionally to the n_trails parameter, it also takes a walltime_limit parameter 
        that defines the maximum time in seconds that the search will take.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to cross validate on
        scoring : str or callable (custom score)
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`
        n_trails : int
            max number of parameter sets to test
        cv_num : int
            number of different random splits
        small_data_eval : bool
            if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
        walltime_limit : int
            the maximum time in seconds that SMAC is allowed to run
        log_level : int
            10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR, 50 - CRITICAL (SMAC3 library log levels)
        \*\*kwargs:
            additional parameters from child-class for ``cross validation`` methods

        Returns
        -------
        incumbent : ConfigSpace.Configuration
            ConfigSpace.Configuration with best hyperparameters (can be used like dict)
        """
        if not SMAC_INSTALLED:
            raise ImportError("SMAC3 library is not installed -> follow instructions in Repo to install SMAC3 (https://github.com/Priapos1004/SAM_ML)")

        logger.debug("starting smac_search")
        # NormalInteger in grid is not supported (using workaround for now) (04/07/2023)
        if self.model_type in ("RFC", "ETC", "GBM", "XGBC")+("RFR", "ETR", "XGBR"):
            grid = self.smac_grid
        else:
            grid = self.grid

        scenario = Scenario(
            grid,
            n_trials=n_trails,
            deterministic=True,
            walltime_limit=walltime_limit,
        )

        initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)
        logger.debug(f"initial_design: {initial_design.select_configurations()}")

        # custom scoring
        if inspect.isfunction(scoring):
            custom_score = scoring
            scoring = "custom_score"
        else:
            custom_score = None

        # define target function
        def grid_train(config: Configuration, seed: int) -> float:
            logger.debug(f"config: {config}")
            model = self.get_deepcopy()
            model.set_params(**config)
            if small_data_eval:
                score = model.cross_validation_small_data(x_train, y_train, console_out=False, leave_loadbar=False, **kwargs, custom_score=custom_score)
            else:
                score = model.cross_validation(x_train, y_train, console_out=False, cv_num=cv_num, **kwargs, custom_score=custom_score)
            return 1 - score[scoring]  # SMAC always minimizes (the smaller the better)

        # use SMAC to find the best hyperparameters
        smac = HyperparameterOptimizationFacade(
            scenario,
            grid_train,
            initial_design=initial_design,
            overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
            logging_level=log_level,
        )

        incumbent = smac.optimize()
        logger.debug("finished smac_search")
        return incumbent
    
    def randomCVsearch(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_trails: int,
        cv_num: int,
        scoring: str | Callable,
        small_data_eval: bool,
        leave_loadbar: bool,
        **kwargs,
    ) -> tuple[dict, float]:
        """
        Hyperparametertuning with randomCVsearch

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to cross validate on
        n_trails : int
            max number of parameter sets to test
        cv_num : int
            number of different random splits
        scoring : str or callable (custom score)
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`
        small_data_eval : bool
            if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
        leave_loadbar : bool
            shall the loading bar of the different parameter sets be visible after training (True - load bar will still be visible)
        \*\*kwargs:
            additional parameters from child-class for ``cross validation`` methods

        Returns
        -------
        best_hyperparameters : dict
            best hyperparameter set
        best_score : float
            the score of the best hyperparameter set

        Notes
        -----
        if you interrupt the keyboard during the run of randomCVsearch, the interim result will be returned
        """
        logger.debug("starting randomCVsearch")
        results = []
        configs = self.get_random_configs(n_trails)

        # custom scoring
        if inspect.isfunction(scoring):
            custom_score = scoring
            scoring = "custom_score"
        else:
            custom_score = None

        at_least_one_run: bool = False
        try:
            for config in tqdm(configs, desc=f"randomCVsearch ({self.model_name})", leave=leave_loadbar):
                logger.debug(f"config: {config}")
                model = self.get_deepcopy()
                model.set_params(**config)
                if small_data_eval:
                    score = model.cross_validation_small_data(x_train, y_train, console_out=False, leave_loadbar=False, **kwargs, custom_score=custom_score)
                else:
                    score = model.cross_validation(x_train, y_train, cv_num=cv_num, console_out=False, **kwargs, custom_score=custom_score)
                config_dict = dict(config)
                config_dict[scoring] = score[scoring]
                results.append(config_dict)
                at_least_one_run = True
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt - output interim result")
            if not at_least_one_run:
                return {}, -1
            

        self._rCVsearch_results = pd.DataFrame(results, dtype=object).sort_values(by=scoring, ascending=False)

        # for-loop to keep dtypes of columns
        best_hyperparameters = {} 
        for col in self._rCVsearch_results.columns:
            value = self._rCVsearch_results[col].iloc[0]
            if str(value) != "nan":
                best_hyperparameters[col] = value

        best_score = best_hyperparameters[scoring]
        best_hyperparameters.pop(scoring)

        logger.debug("finished randomCVsearch")
        
        return best_hyperparameters, best_score
    
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
        only_estimator : bool, default=False
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
