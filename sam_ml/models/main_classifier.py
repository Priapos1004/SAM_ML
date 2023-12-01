import inspect
import os
import sys
import warnings
from datetime import timedelta
from inspect import isfunction
from statistics import mean
from typing import Callable, Literal

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from matplotlib import pyplot as plt
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate
from tqdm.auto import tqdm

from sam_ml.config import (
    get_avg,
    get_n_jobs,
    get_pos_label,
    get_scoring,
    get_secondary_scoring,
    get_strength,
    setup_logger,
)

from .main_model import Model
from .scorer import l_scoring, s_scoring

SMAC_INSTALLED: bool
try:
    from smac import HyperparameterOptimizationFacade, Scenario
    SMAC_INSTALLED = True
except:
    SMAC_INSTALLED = False

logger = setup_logger(__name__)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affects subprocesses


class Classifier(Model):
    """ Classifier parent class {abstract} """

    def __init__(self, model_object, model_name: str, model_type: str, grid: ConfigurationSpace):
        """
        Parameters
        ----------
        model_object : classifier object
            model with 'fit', 'predict', 'set_params', and 'get_params' method (see sklearn API)
        model_name : str
            name of the model
        model_type : str
            kind of estimator (e.g. 'RFC' for RandomForestClassifier)
        grid : ConfigurationSpace
            hyperparameter grid for the model
        """
        super().__init__(model_object, model_name, model_type)
        self._grid = grid
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
    def grid(self) -> ConfigurationSpace:
        """
        Returns
        -------
        grid : ConfigurationSpace
            hyperparameter tuning grid of the model
        """
        return self._grid
    
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
    
    def get_random_configs(self, n_trails: int) -> list:
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

    def replace_grid(self, new_grid: ConfigurationSpace):
        """
        Function to replace self.grid

        Parameters
        ----------
        new_grid : ConfigurationSpace
            new grid to replace the old one with

        Returns
        -------
        changes self.grid variable

        Examples
        --------
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

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series, 
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        console_out: bool = True
    ) -> tuple[float, str]:
        """
        Function to train the model

        Every classifier has a train- and fit-method. They both use the fit-method of the wrapped model, 
        but the train-method returns the train time and the train score of the model. 

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train model
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), \
                default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        console_out : bool, \
                default=True
            shall the score and time be printed out

        Returns
        -------
        train_score : float 
            train score value
        train_time : str
            train time in format: "0:00:00" (hours:minutes:seconds)

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>>
        >>> # train model
        >>> from sam_ml.models.classifier import LR
        >>> 
        >>> model = LR()
        >>> model.train(X, y)
        Train score: 0.9891840171120917 - Train time: 0:00:02
        """
        return super().train(x_train, y_train, console_out, scoring=scoring, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
    
    def train_warm_start(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series, 
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        console_out: bool = True
    ) -> tuple[float, str]:
        """
        Function to warm_start train the model

        This function only differs for pipeline objects (with preprocessing) from the train method.
        For pipeline objects, it only traines the preprocessing steps the first time and then only uses them to preprocess.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train model
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), \
                default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        console_out : bool, \
                default=True
            shall the score and time be printed out

        Returns
        -------
        train_score : float 
            train score value
        train_time : str
            train time in format: "0:00:00" (hours:minutes:seconds)

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>>
        >>> # train model
        >>> from sam_ml.models.classifier import LR
        >>> 
        >>> model = LR()
        >>> model.train_warm_start(X, y)
        Train score: 0.9891840171120917 - Train time: 0:00:02
        """
        return super().train_warm_start(x_train, y_train, console_out, scoring=scoring, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
    
    def __get_score(
        self,
        scoring: str,
        y_test: pd.Series,
        pred: list,
        avg: str,
        pos_label: int | str,
        secondary_scoring: str | None,
        strength: int,
    ) -> float:
        """ 
        Calculate a score for given y true and y prediction values

        Parameters
        ----------
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), \
                default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        y_test, pred : pd.Series, pd.Series
            Data to evaluate model
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")

        Returns
        -------
        score : float 
            metrics score value
        """
        if scoring == "accuracy":
            score = accuracy_score(y_test, pred)
        elif scoring == "precision":
            score = precision_score(y_test, pred, average=avg, pos_label=pos_label)
        elif scoring == "recall":
            score = recall_score(y_test, pred, average=avg, pos_label=pos_label)
        elif scoring == "s_score":
            score = s_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        elif scoring == "l_score":
            score = l_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        elif isfunction(scoring):
            score = scoring(y_test, pred)
        else:
            raise ValueError(f"scoring='{scoring}' is not supported -> only  'accuracy', 'precision', 'recall', 's_score', or 'l_score'")

        return score
    
    def __get_all_scores(
        self,
        y_test: pd.Series,
        pred: list,
        avg: str,
        pos_label: int | str,
        secondary_scoring: str | None,
        strength: int,
        custom_score: Callable[[list[int], list[int]], float] | None,
    ) -> dict[float]:
        """ 
        Calculate accuracy, precision, recall, s_score, l_score, and optional custom_score metrics

        Parameters
        ----------
        y_test, pred : pd.Series, pd.Series
            Data to evaluate model
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        custom_score : callable, \
                default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...}

            or if ``custom_score != None``:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'custom_score': ...,}
        """
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average=avg, pos_label=pos_label)
        recall = recall_score(y_test, pred, average=avg, pos_label=pos_label)
        s_score = s_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        l_score = l_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)

        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "s_score": s_score,
            "l_score": l_score,
        }

        if isfunction(custom_score):
            custom_scores = custom_score(y_test, pred)
            scores["custom_score"] = custom_scores

        return scores

    def evaluate(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        console_out: bool = True,
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        custom_score: Callable[[list[int], list[int]], float] | None = None,
    ) -> dict[str, float]:
        """
        Function to create multiple scores with predict function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate model
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        console_out : bool, \
                default=True
            shall the result of the different scores and a classification_report be printed into the console
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        custom_score : callable, \
                default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...}

            or if ``custom_score != None``:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'custom_score': ...,}

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
        >>> # train and evaluate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> model.train(x_train, y_train)
        >>> scores = model.evaluate(x_test, y_test)
        accuracy: 0.802
        precision: 0.8030604133545309
        recall: 0.7957575757575757
        s_score: 0.9395778023942218
        l_score: 0.9990945415060262
        <BLANKLINE>
        classification report: 
                        precision   recall  f1-score    support
        <BLANKLINE>
                0       0.81        0.73    0.77        225
                1       0.80        0.86    0.83        275
        <BLANKLINE>
        accuracy                            0.80        500
        macro avg       0.80        0.80    0.80        500
        weighted avg    0.80        0.80    0.80        500
        <BLANKLINE>
        """
        pred = self.predict(x_test)
        scores = self.__get_all_scores(y_test, pred, avg, pos_label, secondary_scoring, strength, custom_score)

        if console_out:
            for key in scores:
                print(f"{key}: {scores[key]}")
            print()
            print("classification report:")
            print(classification_report(y_test, pred))

        return scores
    
    def evaluate_proba(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        console_out: bool = True,
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        custom_score: Callable[[list[int], list[int]], float] | None = None,
        probability: float = 0.5,
    ) -> dict[str, float]:
        """
        Function to create multiple scores for binary classification with predict_proba function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate model
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        console_out : bool, \
                default=True
            shall the result of the different scores and a classification_report be printed
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        custom_score : callable, \
                default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.
        probability: float (0 to 1), \
                default=0.5
            probability for class 1 (with value 0.5 is like evaluate_score function). With increasing the probability parameter, precision will likely increase and recall will decrease (with decreasing the probability parameter, the otherway around).

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...}

            or if ``custom_score != None``:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'custom_score': ...,}

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
        >>> # train and evaluate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> model.train(x_train, y_train)
        >>> scores = model.evaluate_proba(x_test, y_test, probability=0.4)
        accuracy: 0.802
        precision: 0.8030604133545309
        recall: 0.7957575757575757
        s_score: 0.9395778023942218
        l_score: 0.9990945415060262
        <BLANKLINE>
        classification report: 
                        precision   recall  f1-score    support
        <BLANKLINE>
                0       0.81        0.73    0.77        225
                1       0.80        0.86    0.83        275
        <BLANKLINE>
        accuracy                            0.80        500
        macro avg       0.80        0.80    0.80        500
        weighted avg    0.80        0.80    0.80        500
        <BLANKLINE>
        """
        if len(set(y_test)) != 2:
            raise ValueError(f"Expected binary classification data, but received y_test with {len(set(y_test))} classes")

        pred_proba = self.predict_proba(x_test)
        pred = [int(x > probability) for x in pred_proba[:, 1]]
        scores = self.__get_all_scores(y_test, pred, avg, pos_label, secondary_scoring, strength, custom_score)

        if console_out:
            for key in scores:
                print(f"{key}: {scores[key]}")
            print()
            print("classification report:")
            print(classification_report(y_test, pred))

        return scores
    
    def evaluate_score(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
    ) -> float:
        """
        Function to create a score with predict function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate model
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), \
                default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")

        Returns
        -------
        score : float 
            metrics score value

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
        >>> # train and evaluate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> model.train(x_train, y_train)
        >>> recall = model.evaluate_score(x_test, y_test, scoring="recall")
        >>> print(f"recall: {recall}")
        recall: 0.4
        """
        pred = self.predict(x_test)
        score = self.__get_score(scoring, y_test, pred, avg, pos_label, secondary_scoring, strength)

        return score
    
    def evaluate_score_proba(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        probability: float = 0.5,
    ) -> float:
        """
        Function to create a score for binary classification with predict_proba function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate model
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), \
                default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        probability: float (0 to 1), \
                default=0.5
            probability for class 1 (with value 0.5 is like evaluate_score function). With increasing the probability parameter, precision will likely increase and recall will decrease (with decreasing the probability parameter, the otherway around).

        Returns
        -------
        score : float 
            metrics score value

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
        >>> # train and evaluate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> model.train(x_train, y_train)
        >>> recall = model.evaluate_score_proba(x_test, y_test, scoring="recall", probability=0.4)
        >>> print(f"recall: {recall}")
        recall: 0.55
        """
        if len(set(y_test)) != 2:
            raise ValueError(f"Expected binary classification data, but received y_test with {len(set(y_test))} classes")

        pred_proba = self.predict_proba(x_test)
        pred = [int(x > probability) for x in pred_proba[:, 1]]
        score = self.__get_score(scoring, y_test, pred, avg, pos_label, secondary_scoring, strength)

        return score

    def cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 10,
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        console_out: bool = True,
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        custom_score: Callable[[list[int], list[int]], float] | None = None,
    ) -> dict[str, float]:
        """
        Random split crossvalidation

        Parameters
        ----------
        X, y : pd.DataFrame, pd.Series
            Data to cross validate on
        cv_num : int, \
                default=10
            number of different random splits
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        console_out : bool, \
                default=True
            shall the result dataframe of the different scores for the different runs be printed
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        custom_score : callable, \
                default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'train_score': ...,
                'train_time': ...,}

            or if ``custom_score != None``:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'train_score': ...,
                'train_time': ...,
                'custom_score': ...,}

        The scores are also saved in ``self.cv_scores``.

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>> 
        >>> # cross validate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> scores = model.cross_validation(X, y, cv_num=3)
        <BLANKLINE>
                                    0         1         2           average
        fit_time                    1.194662  1.295036  1.210156    1.233285
        score_time                  0.167266  0.149569  0.173546    0.163460
        test_precision (macro)      0.779381  0.809037  0.761263    0.783227
        train_precision (macro)     0.951738  0.947397  0.943044    0.947393
        test_recall (macro)         0.774488  0.800144  0.761423    0.778685
        train_recall (macro)        0.948928  0.943901  0.940066    0.944298
        test_accuracy               0.776978  0.803121  0.762305    0.780802
        train_accuracy              0.950180  0.945411  0.941212    0.945601
        test_s_score                0.923052  0.937806  0.917214    0.926024
        train_s_score               0.990794  0.990162  0.989660    0.990206
        test_l_score                0.998393  0.998836  0.998575    0.998602
        train_l_score               1.000000  1.000000  1.000000    1.000000
        """
        logger.debug(f"cross validation {self.model_name} - started")

        precision_scorer = make_scorer(precision_score, average=avg, pos_label=pos_label)
        recall_scorer = make_scorer(recall_score, average=avg, pos_label=pos_label)
        s_scorer = make_scorer(s_scoring, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        l_scorer = make_scorer(l_scoring, strength=strength, scoring=secondary_scoring, pos_label=pos_label)

        if avg == "binary":
            scorer = {
                f"precision ({avg}, label={pos_label})": precision_scorer,
                f"recall ({avg}, label={pos_label})": recall_scorer,
                "accuracy": "accuracy",
                "s_score": s_scorer,
                "l_score": l_scorer,
            }
        else:
            scorer = {
                f"precision ({avg})": precision_scorer,
                f"recall ({avg})": recall_scorer,
                "accuracy": "accuracy",
                "s_score": s_scorer,
                "l_score": l_scorer,
            }            

        if isfunction(custom_score):
            custom_scorer = make_scorer(custom_score)
            scorer["custom_score"] = custom_scorer
        else:
            custom_scorer = None

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

        self._cv_scores = {
            "accuracy": score[list(score.keys())[6]],
            "precision": score[list(score.keys())[2]],
            "recall": score[list(score.keys())[4]],
            "s_score": score[list(score.keys())[8]],
            "l_score": score[list(score.keys())[10]],
            "train_score": score[list(score.keys())[7]],
            "train_time": str(timedelta(seconds = round(score[list(score.keys())[0]]))),
        }

        if isfunction(custom_score):
            self._cv_scores["custom_score"] = score[list(score.keys())[12]]

        logger.debug(f"cross validation {self.model_name} - finished")

        if console_out:
            print()
            print(pd_scores)

        return self._cv_scores

    def cross_validation_small_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        leave_loadbar: bool = True,
        console_out: bool = True,
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        custom_score: Callable[[list[int], list[int]], float] | None = None,
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
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        leave_loadbar : bool, \
                default=True
            shall the loading bar of the training be visible after training (True - load bar will still be visible)
        console_out : bool, \
                default=True
            shall the result of the different scores and a classification_report be printed into the console
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        custom_score : callable, \
                default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'train_score': ...,
                'train_time': ...,}

            or if ``custom_score != None``:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'train_score': ...,
                'train_time': ...,
                'custom_score': ...,}

        The scores are also saved in ``self.cv_scores``.

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>> 
        >>> # cross validate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> scores = model.cross_validation_small_data(X, y)
        accuracy: 0.7
        precision: 0.7747221430607011
        recall: 0.672883787661406
        s_score: 0.40853182756324635
        l_score: 0.7812935895658734
        train_score: 0.9946286670687757
        train_time: 0:00:00
        <BLANKLINE>
        classification report:
                        precision   recall  f1-score    support
        <BLANKLINE>
                0       0.65        0.96    0.78        82
                1       0.90        0.38    0.54        68
        <BLANKLINE>
        accuracy                            0.70        150
        macro avg       0.77        0.67    0.66        150
        weighted avg    0.76        0.70    0.67        150
        <BLANKLINE>
        """
        logger.debug(f"cross validation {self.model_name} - started")

        predictions = []
        true_values = []
        t_scores = []
        t_times = []
        
        for idx in tqdm(X.index, desc=self.model_name, leave=leave_loadbar):
            x_train = X.drop(idx)
            y_train = y.drop(idx)
            x_test = X.loc[[idx]]
            y_test = y.loc[idx]

            train_score, train_time = self.train(x_train, y_train, console_out=False)
            prediction = self.predict(x_test)

            predictions.append(prediction)
            true_values.append(y_test)
            t_scores.append(train_score)
            t_times.append(train_time)

        self._cv_scores = self.__get_all_scores(true_values, predictions, avg, pos_label, secondary_scoring, strength, custom_score)
        avg_train_score = mean(t_scores)
        avg_train_time = str(timedelta(seconds=round(sum(map(lambda f: int(f[0])*3600 + int(f[1])*60 + int(f[2]), map(lambda f: f.split(':'), t_times)))/len(t_times))))

        self._cv_scores.update({
            "train_score": avg_train_score,
            "train_time": avg_train_time,
        })

        if console_out:
            for key in self._cv_scores:
                print(f"{key}: {self._cv_scores[key]}")
            print()
            print("classification report:")
            print(classification_report(true_values, predictions))

        logger.debug(f"cross validation {self.model_name} - finished")

        return self._cv_scores

    def feature_importance(self) -> plt.show:
        """
        Function to generate a matplotlib plot of the top45 feature importance from the model. 
        You can only use the method if you trained your model before.

        Returns
        -------
        plt.show object
        """
        if not self.feature_names:
            raise NotFittedError("You have to first train the classifier before getting the feature importance (with train-method)")

        if self.model_type == "MLPC":
            importances = [np.mean(i) for i in self.model.coefs_[0]]  # MLP Classifier
        elif self.model_type in ("DTC", "RFC", "GBM", "CBC", "ABC", "ETC", "XGBC"):
            importances = self.model.feature_importances_
        elif self.model_type in ("KNC", "GNB", "BNB", "GPC", "QDA", "BC"):
            logger.warning(f"{self.model_type} does not have a feature importance")
            return
        else:
            importances = self.model.coef_[0]  # "normal"

        # top45 features
        feature_importances = pd.Series(importances, index=self.feature_names).sort_values(ascending=False).head(45)

        fig, ax = plt.subplots()
        if self.model_type in ("RFC", "GBM", "ETC"):
            if self.model_type in ("RFC", "ETC"):
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
        n_trails: int = 50,
        cv_num: int = 5,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        small_data_eval: bool = False,
        walltime_limit: float = 600,
        log_level: int = 20,
    ) -> Configuration:
        """
        Hyperparametertuning with SMAC library HyperparameterOptimizationFacade [can only be used in the version with swig]

        The smac_search-method will more "intelligent" search your hyperparameter space than the randomCVsearch and 
        returns the best hyperparameter set. Additionally to the n_trails parameter, it also takes a walltime_limit parameter 
        that defines the maximum time in seconds that the search will take.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to cross validate on
        n_trails : int, \
                default=50
            max number of parameter sets to test
        cv_num : int, \
                default=5
            number of different random splits
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), \
                default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        small_data_eval : bool, \
                default=False
            if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
        walltime_limit : float, \
                default=500
            the maximum time in seconds that SMAC is allowed to run
        log_level : int, \
                default=20
            10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR, 50 - CRITICAL (SMAC3 library log levels)

        Returns
        -------
        incumbent : ConfigSpace.Configuration
            ConfigSpace.Configuration with best hyperparameters (can be used like dict)

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>>
        >>> # initialise model
        >>> from sam_ml.models.classifier import LR
        >>> model = LR()
        >>>
        >>> # use smac_search
        >>> best_hyperparam = model.smac_search(X, y, n_trails=20, cv_num=5, scoring="recall")
        >>> print(f"best hyperparameters: {best_hyperparam}")
        [INFO][abstract_initial_design.py:82] Using `n_configs` and ignoring `n_configs_per_hyperparameter`.
        [INFO][abstract_initial_design.py:147] Using 2 initial design configurations and 0 additional configurations.
        [INFO][abstract_initial_design.py:147] Using 3 initial design configurations and 0 additional configurations.
        [INFO][abstract_intensifier.py:305] Using only one seed for deterministic scenario.
        [INFO][abstract_intensifier.py:515] Added config 12be8a as new incumbent because there are no incumbents yet.
        [INFO][abstract_intensifier.py:590] Added config ce10f4 and rejected config 12be8a as incumbent because it is not better than the incumbents on 1 instances:
        [INFO][abstract_intensifier.py:590] Added config b35335 and rejected config ce10f4 as incumbent because it is not better than the incumbents on 1 instances:
        [INFO][smbo.py:327] Configuration budget is exhausted:
        [INFO][smbo.py:328] --- Remaining wallclock time: 590.5625982284546
        [INFO][smbo.py:329] --- Remaining cpu time: inf
        [INFO][smbo.py:330] --- Remaining trials: 0
        best hyperparameters: Configuration(values={
        'C': 66.7049177605834,
        'penalty': 'l2',
        'solver': 'lbfgs',
        })
        """
        if not SMAC_INSTALLED:
            raise ImportError("SMAC3 library is not installed -> follow instructions in Repo to install SMAC3 (https://github.com/Priapos1004/SAM_ML)")

        logger.debug("starting smac_search")
        # NormalInteger in grid is not supported (using workaround for now) (04/07/2023)
        if self.model_type in ("RFC", "ETC", "GBM", "XGBC"):
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
        if isfunction(scoring):
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
                score = model.cross_validation_small_data(x_train, y_train, console_out=False, leave_loadbar=False, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength, custom_score=custom_score)
            else:
                score = model.cross_validation(x_train, y_train, console_out=False, cv_num=cv_num, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength, custom_score=custom_score)
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
        n_trails: int = 10,
        cv_num: int = 5,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        small_data_eval: bool = False,
        leave_loadbar: bool = True,
    ) -> tuple[dict, float]:
        """
        Hyperparametertuning with randomCVsearch

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to cross validate on
        n_trails : int, \
                default=10
            max number of parameter sets to test
        cv_num : int, \
                default=5
            number of different random splits
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), \
                default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        avg : {"micro", "macro", "binary", "weighted"} or None, \
                default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, \
                default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, \
                default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, \
                default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        small_data_eval : bool, \
                default=False
            if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
        leave_loadbar : bool, \
                default=True
            shall the loading bar of the different parameter sets be visible after training (True - load bar will still be visible)

        Returns
        -------
        best_hyperparameters : dict
            best hyperparameter set
        best_score : float
            the score of the best hyperparameter set

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>>
        >>> # initialise model
        >>> from sam_ml.models.classifier import LR
        >>> model = LR()
        >>>
        >>> # use randomCVsearch
        >>> best_hyperparam, best_score = model.randomCVsearch(X, y, n_trails=20, cv_num=5, scoring="recall")
        >>> print(f"best hyperparameters: {best_hyperparam}, best score: {best_score}")
        best hyperparameters: {'C': 8.471801418819979, 'penalty': 'l2', 'solver': 'newton-cg'}, best score: 0.765
        """
        logger.debug("starting randomCVsearch")
        results = []
        configs = self.get_random_configs(n_trails)

        # custom scoring
        if isfunction(scoring):
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
                    score = model.cross_validation_small_data(x_train, y_train, console_out=False, leave_loadbar=False, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength, custom_score=custom_score)
                else:
                    score = model.cross_validation(x_train, y_train, cv_num=cv_num, console_out=False, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength, custom_score=custom_score)
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
