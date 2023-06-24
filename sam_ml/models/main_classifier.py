from copy import copy
from datetime import timedelta
from statistics import mean
from typing import Union

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_validate)
# from smac import (HyperparameterOptimizationFacade, MultiFidelityFacade,
#                   RandomFacade, Scenario)
from tqdm.auto import tqdm

from sam_ml.config import setup_logger

from .main_model import Model
from .scorer import l_scoring, s_scoring

logger = setup_logger(__name__)


class Classifier(Model):
    """ Classifier parent class """

    def __init__(self, model_object = None, model_name: str = "classifier", model_type: str = "Classifier", grid: dict[str, list] = {}, is_pipeline: bool = False):
        """
        @params:
            model_object: model with 'fit' and 'predict' method
            model_name: name of the model
            model_type: kind of estimator (e.g. 'RFC' for RandomForestClassifier)
            grid: hyperparameter grid for the model
        """
        super().__init__(model_object, model_name, model_type)
        self._grid = grid
        self.is_pipeline = is_pipeline
        self.cv_scores: dict[str, float] = {}
        self.rCVsearch_results: pd.DataFrame|None = None

    def __repr__(self) -> str:
        params: str = ""
        param_dict = self.get_params(False)
        for key in param_dict:
            if type(param_dict[key]) == str:
                params+= key+"='"+str(param_dict[key])+"', "
            else:
                params+= key+"="+str(param_dict[key])+", "
        params += f"model_name='{self.model_name}'"

        return f"{self.model_type}({params})"

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

    def evaluate(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        avg: str = None,
        pos_label: Union[int, str] = -1,
        console_out: bool = True,
        secondary_scoring: str = None,
        strength: int = 3,
    ) -> dict[str, float]:
        """
        @param:
            x_test, y_test: Data to evaluate model

            avg: average to use for precision and recall score (e.g. "micro", None, "weighted", "binary")
            pos_label: if avg="binary", pos_label says which class to score. pos_label is used by s_score/l_score

            console_out: shall the result be printed into the console

            secondary_scoring: weights the scoring (only for 's_score'/'l_score')
            strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for 's_score'/'l_score')
        """
        pred = self.predict(x_test)

        # Calculate Accuracy, Precision and Recall Metrics
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average=avg, pos_label=pos_label)
        recall = recall_score(y_test, pred, average=avg, pos_label=pos_label)
        s_score = s_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        l_score = l_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)

        if console_out:
            print("accuracy: ", accuracy)
            print("precision: ", precision)
            print("recall: ", recall)
            print("s_score: ", s_score)
            print("l_score: ", l_score)
            print()
            print("classification report: ")
            print(classification_report(y_test, pred))

        self.test_score = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "s_score": s_score,
            "l_score": l_score,
        }

        return self.test_score

    def cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 10,
        avg: str = "macro",
        pos_label: Union[int, str] = -1,
        console_out: bool = True,
        secondary_scoring: str = None,
        strength: int = 3,
    ) -> Union[dict[str, float], pd.DataFrame]:
        """
        @param:
            X, y: data to cross validate on
            cv_num: number of different splits

            avg: average to use for precision and recall score (e.g. "micro", "weighted", "binary")
            pos_label: if avg="binary", pos_label says which class to score. pos_label is used by s_score/l_score

            console_out: shall the result be printed into the console

            secondary_scoring: weights the scoring (only for 's_score'/'l_score')
            strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for 's_score'/'l_score')

        @return:
            dictionary with "accuracy", "precision", "recall", "s_score", "l_score", "avg train score", "avg train time"
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

        cv_scores = cross_validate(
            self,
            X,
            y,
            scoring=scorer,
            cv=cv_num,
            return_train_score=True,
            n_jobs=-1,
        )

        pd_scores = pd.DataFrame(cv_scores).transpose()
        pd_scores["average"] = pd_scores.mean(numeric_only=True, axis=1)

        score = pd_scores["average"]
        self.cv_scores = {
            "accuracy": score[list(score.keys())[6]],
            "precision": score[list(score.keys())[2]],
            "recall": score[list(score.keys())[4]],
            "s_score": score[list(score.keys())[8]],
            "l_score": score[list(score.keys())[10]],
            "train_score": score[list(score.keys())[7]],
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
        avg: str = "macro",
        pos_label: Union[int, str] = -1,
        leave_loadbar: bool = True,
        console_out: bool = True,
        secondary_scoring: str = None,
        strength: int = 3,
    ) -> dict[str, float]:
        """
        Cross validation for small datasets (recommended for datasets with less than 150 datapoints)

        @param:
            X, y: data to cross validate on

            avg: average to use for precision and recall score (e.g. "micro", "weighted", "binary")
            pos_label: if avg="binary", pos_label says which class to score. pos_label is used by s_score/l_score

            leave_loadbar: shall the loading bar of the training be visible after training (True - load bar will still be visible)
            console_out: shall the result be printed into the console

            secondary_scoring: weights the scoring (only for 's_score'/'l_score')
            strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for 's_score'/'l_score')

        @return:
            dictionary with "accuracy", "precision", "recall", "s_score", "l_score", "avg train score", "avg train time"
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

        accuracy = accuracy_score(true_values, predictions)
        precision = precision_score(true_values, predictions, average=avg, pos_label=pos_label)
        recall = recall_score(true_values, predictions, average=avg, pos_label=pos_label)
        s_score = s_scoring(true_values, predictions, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        l_score = l_scoring(true_values, predictions, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        avg_train_score = mean(t_scores)
        avg_train_time = str(timedelta(seconds=round(sum(map(lambda f: int(f[0])*3600 + int(f[1])*60 + int(f[2]), map(lambda f: f.split(':'), t_times)))/len(t_times))))

        self.cv_scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "s_score": s_score,
            "l_score": l_score,
            "train_score": avg_train_score,
            "train_time": avg_train_time,
        }

        logger.debug(f"cross validation {self.model_name} - finished")

        if console_out:
            print()
            print("classification report:")
            print(classification_report(true_values, predictions))

        return self.cv_scores

    def feature_importance(self) -> plt.show:
        """
        feature_importance() generates a matplotlib plot of the feature importance from self.model
        """
        if not self.trained:
            logger.error("You have to first train the classifier before getting the feature importance")
            return

        if self.model_type == "MLPC":
            importances = [np.mean(i) for i in self.model.coefs_[0]]  # MLP Classifier
        elif self.model_type in ("DTC", "RFC", "GBM", "CBC", "ABC", "ETC", "XGBC"):
            importances = self.model.feature_importances_
        elif self.model_type in ("KNC", "GNB", "BNB", "GPC", "QDA", "BC"):
            logger.warning(f"{self.model_name} does not have a feature importance")
            return
        else:
            importances = self.model.coef_[0]  # "normal"

        feature_importances = pd.Series(importances, index=self.feature_names)

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
    
    # def smac_search(
    #         self,
    #         x_train: pd.DataFrame, 
    #         y_train: pd.Series
    # ):
    #     # Next, we create an object, holding general information about the run
    #     scenario = Scenario(
    #         self.grid,
    #         n_trials=5,  # We want to run max 50 trials (combination of config and seed)
    #         deterministic=True,
    #         min_budget=5,
    #         max_budget=25,
    #     )

    #     # We want to run the facade's default initial design, but we want to change the number
    #     # of initial configs to 5.
    #     initial_design = MultiFidelityFacade.get_initial_design(scenario)

    #     # define target function
    #     def grid_train(config: Configuration, seed: int, budget) -> float:
    #         print(config)
    #         print(seed)
    #         print(budget)
    #         params = self.get_params()
    #         #print(params)
    #         params.update(config)
    #         #print(params)
    #         model = type(self)(**params)
    #         score = model.cross_validation(x_train, y_train, console_out=False, cv_num=2)
    #         print(1 - score["accuracy"])
    #         return 1 - score["accuracy"]  # SMAC always minimizes (the smaller the better)

    #     # Now we use SMAC to find the best hyperparameters
    #     smac = MultiFidelityFacade(
    #         scenario,
    #         grid_train,
    #         initial_design=initial_design,
    #         overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
    #     )

    #     incumbent = smac.optimize()

    #     # Get cost of default configuration
    #     default_cost = smac.validate(self.grid.get_default_configuration())
    #     print(f"Default cost: {default_cost}")

    #     # Let's calculate the cost of the incumbent
    #     incumbent_cost = smac.validate(incumbent)
    #     print(f"Incumbent cost: {incumbent_cost}")

    def randomCVsearch(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_trails: int = 10,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = -1,
        secondary_scoring: str = None,
        strength: int = 3,
        small_data_eval: bool = False,
        cv_num: int = 5,
        leave_loadbar: bool = True,
    ) -> tuple[dict, float]:
        results = []
        configs = self.get_random_configs(n_trails)
        at_least_one_run: bool = False
        try:
            for config in tqdm(configs, desc=f"randomCVsearch ({self.model_name})", leave=leave_loadbar):
                model = copy(self)
                model.set_params(**config)
                if small_data_eval:
                    score = model.cross_validation_small_data(x_train, y_train, console_out=False, leave_loadbar=False, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
                else:
                    score = model.cross_validation(x_train, y_train, cv_num=cv_num, console_out=False, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
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
        
        return best_hyperparameters, best_score

    # def gridsearch(
    #     self,
    #     x_train: pd.DataFrame,
    #     y_train: pd.Series,
    #     grid: dict = None,
    #     scoring: str = "accuracy",
    #     avg: str = "macro",
    #     pos_label: Union[int, str] = -1,
    #     cv_num: int = 10,
    #     verbose: int = 0,
    #     rand_search: bool = True,
    #     n_iter_num: int = 75,
    #     console_out: bool = True,
    #     train_afterwards: bool = True,
    #     secondary_scoring: str = None,
    #     strength: int = 3,
    # ):
    #     """
    #     @param:
    #         x_train: DataFrame with train features
    #         y_train: Series with labels

    #         grid: dictonary of parameters to tune (default: default parameter dictionary self.grid)

    #         scoring: metrics to evaluate the models
    #         avg: average to use for precision and recall score (e.g. "micro", "weighted", "binary")
    #         pos_label: if avg="binary", pos_label says which class to score. Else pos_label is ignored (except scoring='s_score'/'l_score')

    #         rand_search: True: RandomizedSearchCV, False: GridSearchCV
    #         n_iter_num: Combinations to try out if rand_search=True

    #         cv_num: number of different splits

    #         verbose: log level (higher number --> more logs)
    #         console_out: output the the results of the different iterations
    #         train_afterwards: train the best model after finding it

    #         secondary_scoring: weights the scoring (only for scoring='s_score'/'l_score')
    #         strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for scoring='s_score'/'l_score')

    #     @return:
    #         set self.model = best model from search
    #     """
    #     if grid is None:
    #         grid = self.grid

    #     if console_out:
    #         print()
    #         print("grid: ", grid)
    #         print()

    #     if scoring == "precision":
    #         scoring = make_scorer(precision_score, average=avg, pos_label=pos_label)
    #     elif scoring == "recall":
    #         scoring = make_scorer(recall_score, average=avg, pos_label=pos_label)
    #     elif scoring == "s_score":
    #         scoring = make_scorer(s_scoring, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
    #     elif scoring == "l_score":
    #         scoring = make_scorer(l_scoring, strength=strength, scoring=secondary_scoring, pos_label=pos_label)

    #     if rand_search:
    #         grid_search = RandomizedSearchCV(
    #             estimator=self,
    #             param_distributions=grid,
    #             n_iter=n_iter_num,
    #             cv=cv_num,
    #             verbose=verbose,
    #             random_state=42,
    #             n_jobs=-1,
    #             scoring=scoring,
    #         )
    #     else:
    #         grid_search = GridSearchCV(
    #             estimator=self,
    #             param_grid=grid,
    #             n_jobs=-1,
    #             cv=cv_num,
    #             verbose=verbose,
    #             scoring=scoring,
    #             error_score=0,
    #         )
    #     logger.debug(f"hyperparameter tuning {self.model_name} - started")
    #     grid_result = grid_search.fit(x_train, y_train)
    #     logger.debug(f"hyperparameter tuning {self.model_name} - finished")

    #     if console_out:
    #         means = grid_result.cv_results_["mean_test_score"]
    #         stds = grid_result.cv_results_["std_test_score"]
    #         params = grid_result.cv_results_["params"]
    #         print()
    #         for mean, stdev, param in zip(means, stds, params):
    #             print("mean: %f (stdev: %f) with: %r" % (mean, stdev, param))
    #         print()

    #     self.model = grid_result.best_estimator_.model
    #     if self.is_pipeline:
    #         self.vectorizer = grid_result.best_estimator_.vectorizer
    #         self.scaler = grid_result.best_estimator_.scaler
    #         self.selector = grid_result.best_estimator_.selector
    #         self.sampler = grid_result.best_estimator_.sampler
    #         self._classifier = (self.model, self.model_type, self._grid)

    #     print()
    #     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #     print()

    #     if train_afterwards:
    #         logger.debug(f"best model training {self.model_name} - started")
    #         self.train(x_train, y_train, console_out=False)
    #         logger.debug(f"best model training {self.model_name} - finished")
