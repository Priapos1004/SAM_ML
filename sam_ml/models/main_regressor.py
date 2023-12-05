import os
import sys
import warnings
from datetime import timedelta
from inspect import isfunction
from typing import Callable, Literal

import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn.metrics import d2_tweedie_score, make_scorer, mean_squared_error, r2_score

from sam_ml.config import setup_logger

from .main_model import Model

logger = setup_logger(__name__)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affects subprocesses


class Regressor(Model):
    """ Regressor parent class """

    def __init__(self, model_object, model_name: str, model_type: str, grid: ConfigurationSpace):
        """
        Parameters
        ----------
        model_object : classifier object
            model with 'fit', 'predict', 'set_params', and 'get_params' method (see sklearn API)
        model_name : str
            name of the model
        model_type : str
            kind of estimator (e.g. 'RFR' for RandomForestRegressor)
        grid : ConfigurationSpace
            hyperparameter grid for the model
        """
        super().__init__(model_object, model_name, model_type, grid)

    def _get_score(
        self,
        scoring: str,
        y_test: pd.Series,
        pred: list,
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

        Returns
        -------
        score : float 
            metrics score value
        """
        if scoring == "r2":
            score = r2_score(y_test, pred)
        elif scoring == "rmse":
            score = mean_squared_error(y_test, pred, squared=False)
        elif scoring == "d2_tweedie":
            if all([y >= 0 for y in y_test]) and all([y > 0 for y in pred]):
                score = d2_tweedie_score(y_test, pred, power=1)
            else:
                logger.warning("There are y_test values smaller 0 or y_pred values smaller-equal 0 -> d2_tweedie_score will be -1")
                score = -1
        elif isfunction(scoring):
            score = scoring(y_test, pred)
        else:
            raise ValueError(f"scoring='{scoring}' is not supported -> only  'r2', 'rmse', or 'd2_tweedie'")

        return score
    
    def _get_all_scores(
        self,
        y_test: pd.Series,
        pred: list,
        custom_score: Callable[[list[int], list[int]], float] | None,
    ) -> dict[str, float]:
        """ 
        Calculate r2, rmse, d2_tweedie, and optional custom_score metrics

        Parameters
        ----------
        y_test, pred : pd.Series, pd.Series
            Data to evaluate model
        custom_score : callable, \
                default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'r2': ...,
                'rmse': ...,
                'd2_tweedie': ...,}

            or if ``custom_score != None``:

                {'r2': ...,
                'rmse': ...,
                'd2_tweedie': ...,
                'custom_score': ...,}

        Notes
        -----
        d2_tweedie is only defined for y_test >= 0 and y_pred > 0 values. Otherwise, d2_tweedie is set to -1.
        """
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred, squared=False)
        if all([y >= 0 for y in y_test]) and all([y > 0 for y in pred]):
            d2_tweedie = d2_tweedie_score(y_test, pred, power=1)
        else:
            d2_tweedie = -1

        scores = {
            "r2": r2,
            "rmse": rmse,
            "d2_tweedie": d2_tweedie,
        }

        if isfunction(custom_score):
            custom_scores = custom_score(y_test, pred)
            scores["custom_score"] = custom_scores

        return scores
    
    def _make_scorer(
        self,
        y_values: pd.Series,
        custom_score: Callable | None,
    ) -> dict[str, Callable]:
        """
        Function to create a dictionary with scorer for the crossvalidation
        
        Parameters
        ----------
        y_values : pd.Series
            y data for testing if d2_tweedie is allowed
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scorer : dict[str, Callable]
            dictionary with scorer functions
        """
        r2 = make_scorer(r2_score)
        rmse = make_scorer(mean_squared_error, squared=False)

        if all([y_elem >= 0 for y_elem in y_values]):
            d2_tweedie = make_scorer(d2_tweedie_score, power=1)
            scorer = {
                "r2 score": r2,
                "rmse": rmse,
                "d2 tweedie score": d2_tweedie,
            }
        else:
            scorer = {
                "r2 score": r2,
                "rmse": rmse,
            }

        if isfunction(custom_score):
            scorer["custom_score"] = make_scorer(custom_score)

        return scorer
    
    def _make_cv_scores(
            self,
            score: dict,
            custom_score: Callable | None = None,
    ) -> dict[str, float]:
        """
        Function to create from the crossvalidation results a dictionary
        
        Parameters
        ----------
        score : dict
            crossvalidation average column results
        custom_score : callable or None, \
                default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        cv_scores : dict
            restructured dictionary
        """
        if len(score)==10:
            cv_scores = {
                "r2": score[list(score.keys())[2]],
                "rmse": score[list(score.keys())[4]],
                "d2_tweedie": score[list(score.keys())[6]],
                "train_score": score[list(score.keys())[3]],
                "train_time": str(timedelta(seconds = round(score[list(score.keys())[0]]))),
            }
            if isfunction(custom_score):
                cv_scores["custom_score"] = score[list(score.keys())[8]]
        else:
            cv_scores = {
                "r2": score[list(score.keys())[2]],
                "rmse": score[list(score.keys())[4]],
                "d2_tweedie": -1,
                "train_score": score[list(score.keys())[3]],
                "train_time": str(timedelta(seconds = round(score[list(score.keys())[0]]))),
            }
            if isfunction(custom_score):
                cv_scores["custom_score"] = score[list(score.keys())[6]]
        
        return cv_scores

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series, 
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[int], list[int]], float] = "r2",
        console_out: bool = True
    ) -> tuple[float, str]:
        """
        @return:
            tuple of train score and train time
        """
        return super().train(x_train, y_train, console_out, scoring=scoring)
    
    def train_warm_start(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series, 
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[int], list[int]], float] = "r2",
        console_out: bool = True
    ) -> tuple[float, str]:
        """
        @return:
            tuple of train score and train time
        """
        return super().train_warm_start(x_train, y_train, console_out, scoring=scoring)

    def evaluate(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        console_out: bool = True,
        custom_score: Callable[[list[float], list[float]], float] | None = None,
    ) -> dict[str, float]:
        """
        @param:
            x_test, y_test: Data to evaluate model

            console_out: shall the result be printed into the console

            custom_score: score function with 'y_true' and 'y_pred' as parameter

        @return: dictionary with keys with scores: "r2", "rmse", "d2_tweedie"
        """
        return super().evaluate(x_test, y_test, console_out=console_out, custom_score=custom_score)
    
    def evaluate_score(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[float], list[float]], float] = "r2",
    ) -> float:
        """
        @param:
            x_test, y_test: Data to evaluate model
            scoring: metrics to evaluate the models ("r2", "rmse", "d2_tweedie", score function)

        @return: score as float
        """
        return super().evaluate_score(scoring, x_test, y_test)

    def cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 10,
        console_out: bool = True,
        custom_score: Callable[[list[float], list[float]], float] | None = None,
    ) -> dict[str, float]:
        """
        @param:
            X, y: data to cross validate on
            cv_num: number of different splits

            console_out: shall the result be printed into the console

            custom_score: score function with 'y_true' and 'y_pred' as parameter

        @return:
            dictionary with "r2", "rmse", "d2_tweedie", "train_score", "train_time"
        """
        return super().cross_validation(X, y, cv_num=cv_num, console_out=console_out, custom_score=custom_score, y_values=y)

    def cross_validation_small_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        leave_loadbar: bool = True,
        console_out: bool = True,
        custom_score: Callable[[list[float], list[float]], float] | None = None,
    ) -> dict[str, float]:
        """
        Cross validation for small datasets (recommended for datasets with less than 150 datapoints)

        @param:
            X, y: data to cross validate on

            leave_loadbar: shall the loading bar of the training be visible after training (True - load bar will still be visible)

            custom_score: score function with 'y_true' and 'y_pred' as parameter
            
        @return:
            dictionary with "r2", "rmse", "d2_tweedie", "train_score", "train_time"
        """
        return super().cross_validation_small_data(X,y,leave_loadbar=leave_loadbar, console_out=console_out, custom_score=custom_score)
    
    def smac_search(
        self,
        x_train: pd.DataFrame, 
        y_train: pd.Series,
        n_trails: int = 50,
        cv_num: int = 5,
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[float], list[float]], float] = "r2",
        small_data_eval: bool = False,
        walltime_limit: float = 600,
        log_level: int = 20,
    ) -> Configuration:
        """
        @params:
            x_train: DataFrame with train features
            y_train: Series with labels

            n_trails: max number of parameter sets to test
            cv_num: number of different splits per crossvalidation (only used when small_data_eval=False)

            scoring: metrics to evaluate the models ("r2", "rmse", "d2_tweedie", score function)

            small_data_eval: if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
            
            walltime_limit: the maximum time in seconds that SMAC is allowed to run

            log_level: 10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR, 50 - CRITICAL (SMAC3 library log levels)

        @return: ConfigSpace.Configuration with best hyperparameters (can be used like dict)
        """
        return super().smac_search(x_train, y_train, scoring=scoring, n_trails=n_trails, cv_num=cv_num, small_data_eval=small_data_eval, walltime_limit=walltime_limit, log_level=log_level)

    def randomCVsearch(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_trails: int = 10,
        cv_num: int = 5,
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[float], list[float]], float] = "r2",
        small_data_eval: bool = False,
        leave_loadbar: bool = True,
    ) -> tuple[dict, float]:
        """
        @params:
            x_train: DataFrame with train features
            y_train: Series with labels

            n_trails: number of parameter sets to test

            scoring: metrics to evaluate the models ("r2", "rmse", "d2_tweedie", score function)

            small_data_eval: if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)

            cv_num: number of different splits per crossvalidation (only used when small_data_eval=False)

            leave_loadbar: shall the loading bar of the different parameter sets be visible after training (True - load bar will still be visible)

        @return: dictionary with best hyperparameters and float of best_score
        """
        return super().randomCVsearch(x_train, y_train, n_trails=n_trails, cv_num=cv_num, scoring=scoring, small_data_eval=small_data_eval, leave_loadbar=leave_loadbar)
