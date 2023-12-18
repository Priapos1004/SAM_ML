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
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[float], list[float]], float],
        y_test: pd.Series,
        pred: list,
    ) -> float:
        """ 
        Calculate a score for given y true and y prediction values

        Parameters
        ----------
        scoring : {"r2", "rmse", "d2_tweedie"} or callable (custom score)
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
        custom_score: Callable[[list[float], list[float]], float] | None,
    ) -> dict[str, float]:
        """ 
        Calculate r2, rmse, d2_tweedie, and optional custom_score metrics

        Parameters
        ----------
        y_test, pred : pd.Series, pd.Series
            Data to evaluate model
        custom_score : callable or None
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
        custom_score: Callable[[list[float], list[float]], float] | None,
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
            custom_score: Callable[[list[float], list[float]], float] | None,
    ) -> dict[str, float]:
        """
        Function to create from the crossvalidation results a dictionary
        
        Parameters
        ----------
        score : dict
            crossvalidation average column results
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        cv_scores : dict
            restructured dictionary
        """
        if (len(score)==10 and isfunction(custom_score)) or (len(score)==8 and not isfunction(custom_score)):
            cv_scores = {
                "r2": score[list(score.keys())[2]],
                "rmse": score[list(score.keys())[4]],
                "d2_tweedie": score[list(score.keys())[6]],
                "train_time": str(timedelta(seconds = round(score[list(score.keys())[0]]))),
                "train_score": score[list(score.keys())[3]],
            }
            if isfunction(custom_score):
                cv_scores["custom_score"] = score[list(score.keys())[8]]
        else:
            cv_scores = {
                "r2": score[list(score.keys())[2]],
                "rmse": score[list(score.keys())[4]],
                "d2_tweedie": -1,
                "train_time": str(timedelta(seconds = round(score[list(score.keys())[0]]))),
                "train_score": score[list(score.keys())[3]],
            }
            if isfunction(custom_score):
                cv_scores["custom_score"] = score[list(score.keys())[6]]
        
        return cv_scores

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series, 
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[float], list[float]], float] = "r2",
        console_out: bool = True
    ) -> tuple[float, str]:
        """
        Function to train the model

        Every regressor has a train- and fit-method. They both use the fit-method of the wrapped model, 
        but the train-method returns the train time and the train score of the model.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train model
        scoring : {"r2", "rmse", "d2_tweedie"} or callable (custom score), default="r2"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        console_out : bool, default=True
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
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>>
        >>> # train model
        >>> from sam_ml.models.regressor import RFR
        >>> 
        >>> model = RFR()
        >>> model.train(X, y)
        Train score: 0.9938023719617127 - Train time: 0:00:01
        """
        return super().train(x_train, y_train, console_out, scoring=scoring)
    
    def train_warm_start(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series, 
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[float], list[float]], float] = "r2",
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
        scoring : {"r2", "rmse", "d2_tweedie"} or callable (custom score), default="r2"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        console_out : bool, default=True
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
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>>
        >>> # train model
        >>> from sam_ml.models.regressor import RFR
        >>> 
        >>> model = RFR()
        >>> model.train_warm_start(X, y)
        Train score: 0.9938023719617127 - Train time: 0:00:01
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
        Function to create multiple scores with predict function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate model
        console_out : bool, default=True
            shall the result of the different scores and a classification_report be printed into the console
        custom_score : callable or None, default=None
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

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>>
        >>> # train and evaluate model
        >>> from sam_ml.models.regressor import RFR
        >>> 
        >>> model = RFR()
        >>> model.train(X, y)
        >>> scores = model.evaluate(x_test, y_test)
        Train score: 0.9938023719617127 - Train time: 0:00:01
        r2: 0.9471767309072388
        rmse: 11.46914444113609
        d2_tweedie: 0.9214227488752569
        """
        return super().evaluate(x_test, y_test, console_out=console_out, custom_score=custom_score)
    
    def evaluate_score(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[float], list[float]], float] = "r2",
    ) -> float:
        """
        Function to create a score with predict function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate model
        scoring : {"r2", "rmse", "d2_tweedie"} or callable (custom score), default="r2"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

        Returns
        -------
        score : float 
            metrics score value

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>>
        >>> # train and evaluate model
        >>> from sam_ml.models.regressor import RFR
        >>> 
        >>> model = RFR()
        >>> model.fit(X, y)
        >>> rmse = model.evaluate_score(x_test, y_test, scoring="rmse")
        >>> print(f"rmse: {rmse}")
        rmse: 11.46914444113609
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
        Random split crossvalidation

        Parameters
        ----------
        X, y : pd.DataFrame, pd.Series
            Data to cross validate on
        cv_num : int, default=10
            number of different random splits
        console_out : bool, default=True
            shall the result dataframe of the different scores for the different runs be printed
        custom_score : callable or None, default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'r2': ...,
                'rmse': ...,
                'd2_tweedie': ...,
                'train_time': ...,
                'train_score': ...,}

            or if ``custom_score != None``:

                {'r2': ...,
                'rmse': ...,
                'd2_tweedie': ...,
                'train_time': ...,
                'train_score': ...,
                'custom_score': ...,}

        The scores are also saved in ``self.cv_scores``.

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>>
        >>> # cross validate model
        >>> from sam_ml.models.regressor import RFR
        >>> 
        >>> model = RFR()
        >>> scores = model.cross_validation(X, y, cv_num=3)
        <BLANKLINE>
                                    0          1          2         average
        fit_time                    0.772634   0.903580   0.769893  0.815369
        score_time                  0.097742   0.126724   0.108220  0.110895
        test_r2 score               0.930978   0.935554   0.950584  0.939039
        train_r2 score              0.992086   0.992418   0.991672  0.992059
        test_rmse                   13.122513  12.076931  10.936810 12.045418
        train_rmse                  4.306834   4.318027   4.457605  4.360822
        test_d2 tweedie score       0.916618   0.909032   0.919350  0.915000
        train_d2 tweedie score      0.982802   0.983685   0.983286  0.983257
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
        leave_loadbar : bool, default=True
            shall the loading bar of the training be visible after training (True - load bar will still be visible)
        console_out : bool, default=True
            shall the result of the different scores and a classification_report be printed into the console
        custom_score : callable or None, default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'r2': ...,
                'rmse': ...,
                'd2_tweedie': ...,
                'train_time': ...,
                'train_score': ...,}

            or if ``custom_score != None``:

                {'r2': ...,
                'rmse': ...,
                'd2_tweedie': ...,
                'train_time': ...,
                'train_score': ...,
                'custom_score': ...,}

        The scores are also saved in ``self.cv_scores``.

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=150, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>>
        >>> # cross validate model
        >>> from sam_ml.models.regressor import RFR
        >>> 
        >>> model = RFR()
        >>> scores = model.cross_validation_small_data(X, y)
        r2: 0.5914164661854215
        rmse: 50.2870203230133
        d2_tweedie: 0.58636121702529
        train_time: 0:00:00
        train_score: 0.9425178468662095
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
        walltime_limit: int = 600,
        log_level: int = 20,
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
        n_trails : int, default=50
            max number of parameter sets to test
        cv_num : int, default=5
            number of different random splits
        scoring : {"r2", "rmse", "d2_tweedie"} or callable (custom score), default="r2"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        small_data_eval : bool, default=False
            if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
        walltime_limit : int, default=600
            the maximum time in seconds that SMAC is allowed to run
        log_level : int, default=20
            10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR, 50 - CRITICAL (SMAC3 library log levels)

        Returns
        -------
        incumbent : ConfigSpace.Configuration
            ConfigSpace.Configuration with best hyperparameters (can be used like dict)

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>>
        >>> # use smac_search
        >>> from sam_ml.models.regressor import RFR
        >>> 
        >>> model = RFR()
        >>> best_hyperparam = model.smac_search(X, y, n_trails=20, cv_num=5, scoring="rmse")
        >>> print(f"best hyperparameters: {best_hyperparam}")
        [INFO][abstract_initial_design.py:147] Using 5 initial design configurations and 0 additional configurations.
        [INFO][abstract_intensifier.py:305] Using only one seed for deterministic scenario.
        [INFO][abstract_intensifier.py:515] Added config 7373ff as new incumbent because there are no incumbents yet.
        [INFO][abstract_intensifier.py:590] Added config 06e4dc and rejected config 7373ff as incumbent because it is not better than the incumbents on 1 instances:
        [INFO][abstract_intensifier.py:590] Added config 162148 and rejected config 06e4dc as incumbent because it is not better than the incumbents on 1 instances:
        [INFO][abstract_intensifier.py:590] Added config 97eecc and rejected config 162148 as incumbent because it is not better than the incumbents on 1 instances:
        [INFO][smbo.py:327] Configuration budget is exhausted:
        [INFO][smbo.py:328] --- Remaining wallclock time: 582.9456326961517
        [INFO][smbo.py:329] --- Remaining cpu time: inf
        [INFO][smbo.py:330] --- Remaining trials: 0
        best hyperparameters: Configuration(values={
        'bootstrap': False,
        'criterion': 'friedman_mse',
        'max_depth': 10,
        'min_samples_leaf': 3,
        'min_samples_split': 9,
        'min_weight_fraction_leaf': 0.22684614269623157,
        'n_estimators': 28,
        })
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
        Hyperparametertuning with randomCVsearch

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to cross validate on
        n_trails : int, default=10
            max number of parameter sets to test
        cv_num : int, default=5
            number of different random splits
        scoring : {"r2", "rmse", "d2_tweedie"} or callable (custom score), default="r2"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        small_data_eval : bool, default=False
            if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
        leave_loadbar : bool, default=True
            shall the loading bar of the different parameter sets be visible after training (True - load bar will still be visible)

        Returns
        -------
        best_hyperparameters : dict
            best hyperparameter set
        best_score : float
            the score of the best hyperparameter set

        Notes
        -----
        if you interrupt the keyboard during the run of randomCVsearch, the interim result will be returned

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>>
        >>> # use randomCVsearch
        >>> from sam_ml.models.regressor import RFR
        >>> 
        >>> model = RFR()
        >>> best_hyperparam, best_score = model.randomCVsearch(X, y, n_trails=20, cv_num=5, scoring="r2")
        >>> print(f"best hyperparameters: {best_hyperparam}, best score: {best_score}")
        best hyperparameters: {'bootstrap': True, 'criterion': 'friedman_mse', 'max_depth': 9, 'min_samples_leaf': 4, 'min_samples_split': 7, 'min_weight_fraction_leaf': 0.015714592843367126, 'n_estimators': 117}, best score: 0.6880857784416011
        """
        return super().randomCVsearch(x_train, y_train, n_trails=n_trails, cv_num=cv_num, scoring=scoring, small_data_eval=small_data_eval, leave_loadbar=leave_loadbar)
