import inspect
import os
import sys
import warnings
from datetime import timedelta
from statistics import mean

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from matplotlib import pyplot as plt
from sklearn.exceptions import NotFittedError
from sklearn.metrics import d2_tweedie_score, make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from tqdm.auto import tqdm

from sam_ml.config import get_n_jobs, setup_logger

from .main_model import Model

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


class Regressor(Model):
    """ Regressor parent class """

    def __init__(self, model_object = None, model_name: str = "regressor", model_type: str = "Regressor", grid: ConfigurationSpace = ConfigurationSpace()):
        """
        @params:
            model_object: model with 'fit', 'predict', 'set_params', and 'get_params' method (see sklearn API)
            model_name: name of the model
            model_type: kind of estimator (e.g. 'RFR' for RandomForestRegressor)
            grid: hyperparameter grid for the model
        """
        super().__init__(model_object, model_name, model_type)
        self._grid = grid
        self.cv_scores: dict[str, float] = {}
        self.rCVsearch_results: pd.DataFrame|None = None

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
    def grid(self):
        """
        @return:
            hyperparameter tuning grid of the model
        """
        return self._grid
    
    def get_random_config(self):
        """
        @return;
            set of random parameter from grid
        """
        return dict(self.grid.sample_configuration(1))
    
    def get_random_configs(self, n_trails: int) -> list:
        """
        @return;
            n_trails elements in list with sets of random parameterd from grid

        NOTE: filter out duplicates -> could be less than n_trails
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
        function to replace self.grid 

        e.g.:
            ConfigurationSpace(
                seed=42,
                space={
                    "solver": Categorical("solver", ["lsqr", "eigen", "svd"]),
                    "shrinkage": Float("shrinkage", (0, 1)),
                })
        """
        self._grid = new_grid

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series, 
        scoring: str = "r2",
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
        scoring: str = "r2",
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
    ) -> dict[str, float]:
        """
        @param:
            x_test, y_test: Data to evaluate model

            console_out: shall the result be printed into the console

        @return: dictionary with keys with scores: "r2", "rmse", "d2_tweedie"
        """
        pred = self.predict(x_test)

        # Calculate Metrics
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred, squared=False)
        if all([y >= 0 for y in y_test]):
            d2_tweedie = d2_tweedie_score(y_test, pred, power=1)
        else:
            d2_tweedie = -1

        if console_out:
            print("r2 score: ", r2)
            print("rmse: ", rmse)
            print("d2 tweedie score: ", d2_tweedie)

        scores = {
            "r2": r2,
            "rmse": rmse,
            "d2_tweedie": d2_tweedie,
        }

        return scores
    
    def evaluate_score(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: str = "r2",
    ) -> float:
        """
        @param:
            x_test, y_test: Data to evaluate model
            scoring: metrics to evaluate the models ("r2", "rmse", "d2_tweedie")

        @return: score as float
        """
        pred = self.predict(x_test)

        # Calculate score
        if scoring == "r2":
            score = r2_score(y_test, pred)
        elif scoring == "rmse":
            score = mean_squared_error(y_test, pred, squared=False)
        elif scoring == "d2_tweedie":
            if all([y >= 0 for y in y_test]):
                score = d2_tweedie_score(y_test, pred, power=1)
            else:
                logger.warning("There are y values smaller 0 -> d2_tweedie_score will always be -1")
                score = -1
        else:
            raise ValueError(f"scoring='{scoring}' is not supported -> only  'r2', 'rmse', or 'd2_tweedie'")

        return score

    def cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 10,
        console_out: bool = True,
    ) -> dict[str, float]:
        """
        @param:
            X, y: data to cross validate on
            cv_num: number of different splits

            console_out: shall the result be printed into the console

        @return:
            dictionary with "r2", "rmse", "d2_tweedie", "train_score", "train_time"
        """
        logger.debug(f"cross validation {self.model_name} - started")

        r2 = make_scorer(r2_score)
        rmse = make_scorer(mean_squared_error, squared=False)

        if all([y_elem >= 0 for y_elem in y]):
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
        
        if all([y_elem >= 0 for y_elem in y]):
            self.cv_scores = {
                "r2": score[list(score.keys())[2]],
                "rmse": score[list(score.keys())[4]],
                "d2_tweedie": score[list(score.keys())[6]],
                "train_score": score[list(score.keys())[3]],
                "train_time": str(timedelta(seconds = round(score[list(score.keys())[0]]))),
            }
        else:
            self.cv_scores = {
                "r2": score[list(score.keys())[2]],
                "rmse": score[list(score.keys())[4]],
                "d2_tweedie": -1,
                "train_score": score[list(score.keys())[3]],
                "train_time": str(timedelta(seconds = round(score[list(score.keys())[0]]))),
            }

        logger.debug(f"cross validation {self.model_name} - finished")

        if console_out:
            print()
            print(pd_scores)

        return self.cv_scores

    def cross_validation_small_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        leave_loadbar: bool = True,
    ) -> dict[str, float]:
        """
        Cross validation for small datasets (recommended for datasets with less than 150 datapoints)

        @param:
            X, y: data to cross validate on

            leave_loadbar: shall the loading bar of the training be visible after training (True - load bar will still be visible)
            
        @return:
            dictionary with "r2", "rmse", "d2_tweedie", "train_score", "train_time"
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

        # Calculate Metrics
        r2 = r2_score(true_values, predictions)
        rmse = mean_squared_error(true_values, predictions, squared=False)

        if all([y_elem >= 0 for y_elem in y]):
            d2_tweedie = d2_tweedie_score(true_values, predictions, power=1)
        else:
            d2_tweedie = -1
        
        avg_train_score = mean(t_scores)
        avg_train_time = str(timedelta(seconds=round(sum(map(lambda f: int(f[0])*3600 + int(f[1])*60 + int(f[2]), map(lambda f: f.split(':'), t_times)))/len(t_times))))

        self.cv_scores = {
            "r2": r2,
            "rmse": rmse,
            "d2_tweedie": d2_tweedie,
            "train_score": avg_train_score,
            "train_time": avg_train_time,
        }

        logger.debug(f"cross validation {self.model_name} - finished")

        return self.cv_scores

    def feature_importance(self) -> plt.show:
        """
        feature_importance() generates a matplotlib plot of the feature importance from self.model
        """
        if not self.feature_names:
            raise NotFittedError("You have to first train the classifier before getting the feature importance (with train-method)")

        if self.model_type == "...":
            importances = [np.mean(i) for i in self.model.coefs_[0]]  # MLP Regressor
        elif self.model_type in ("RFR", "DTR", "ETR"):
            importances = self.model.feature_importances_
        elif self.model_type in ():
            logger.warning(f"{self.model_type} does not have a feature importance")
            return
        else:
            importances = self.model.coef_[0]  # "normal"

        feature_importances = pd.Series(importances, index=self.feature_names)

        fig, ax = plt.subplots()
        if self.model_type in ("RFR", "ETR"):
            if self.model_type in ("RFR", "ETR"):
                std = np.std(
                    [tree.feature_importances_ for tree in self.model.estimators_], axis=0,
                )
            elif self.model_type == "...":
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
        scoring: str = "r2",
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

            scoring: metrics to evaluate the models ("r2", "rmse", "d2_tweedie")

            small_data_eval: if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
            
            walltime_limit: the maximum time in seconds that SMAC is allowed to run

            log_level: 10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR, 50 - CRITICAL (SMAC3 library log levels)

        @return: ConfigSpace.Configuration with best hyperparameters (can be used like dict)
        """
        if not SMAC_INSTALLED:
            raise ImportError("SMAC3 library is not installed -> follow instructions in Repo to install SMAC3 (https://github.com/Priapos1004/SAM_ML)")

        logger.debug("starting smac_search")
        # NormalInteger in grid is not supported (using workaround for now) (04/07/2023)
        if self.model_type in ("RFR", "ETR"):
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

        # define target function
        def grid_train(config: Configuration, seed: int) -> float:
            logger.debug(f"config: {config}")
            model = self.get_deepcopy()
            model.set_params(**config)
            if small_data_eval:
                score = model.cross_validation_small_data(x_train, y_train, leave_loadbar=False)
            else:
                score = model.cross_validation(x_train, y_train, console_out=False, cv_num=cv_num)
            
            # SMAC always minimizes (the smaller the better)
            if scoring == "rmse":
                return score[scoring]
            
            return 1 - score[scoring]

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
        scoring: str = "r2",
        small_data_eval: bool = False,
        leave_loadbar: bool = True,
    ) -> tuple[dict, float]:
        """
        @params:
            x_train: DataFrame with train features
            y_train: Series with labels

            n_trails: number of parameter sets to test

            scoring: metrics to evaluate the models ("r2", "rmse", "d2_tweedie")

            small_data_eval: if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)

            cv_num: number of different splits per crossvalidation (only used when small_data_eval=False)

            leave_loadbar: shall the loading bar of the different parameter sets be visible after training (True - load bar will still be visible)

        @return: dictionary with best hyperparameters and float of best_score
        """
        logger.debug("starting randomCVsearch")
        results = []
        configs = self.get_random_configs(n_trails)
        at_least_one_run: bool = False
        try:
            for config in tqdm(configs, desc=f"randomCVsearch ({self.model_name})", leave=leave_loadbar):
                logger.debug(f"config: {config}")
                model = self.get_deepcopy()
                model.set_params(**config)
                if small_data_eval:
                    score = model.cross_validation_small_data(x_train, y_train, leave_loadbar=False)
                else:
                    score = model.cross_validation(x_train, y_train, cv_num=cv_num, console_out=False)
                config_dict = dict(config)
                config_dict[scoring] = score[scoring]
                results.append(config_dict)
                at_least_one_run = True
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt - output interim result")
            if not at_least_one_run:
                return {}, -1
            

        self.rCVsearch_results = pd.DataFrame(results, dtype=object).sort_values(by=scoring, ascending=False)

        # for-loop to keep dtypes of columns
        best_hyperparameters = {} 
        for col in self.rCVsearch_results.columns:
            value = self.rCVsearch_results[col].iloc[0]
            if str(value) != "nan":
                best_hyperparameters[col] = value

        best_score = best_hyperparameters[scoring]
        best_hyperparameters.pop(scoring)

        logger.debug("finished randomCVsearch")
        
        return best_hyperparameters, best_score
