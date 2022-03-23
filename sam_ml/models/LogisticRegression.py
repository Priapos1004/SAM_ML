from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from matplotlib import pyplot as plt
from typing import Union
import logging
import pandas as pd
import numpy as np
from .main_classifier import Classifier


class LR(Classifier):
    def __init__(
        self,
        penalty: str = "l2",
        dual: bool = False,
        tol: float = 0.0001,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: Union[str, dict] = None,
        random_state: int = None,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        multi_class: str = "auto",
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: int = -1,
        l1_ratio: float = None,
    ):
        '''
        @param (important one):
            n_jobs - how many cores shall be used (-1 means all)
            random_state - random_state for model
            verbose - log level (higher number --> more logs)
            warm_start - work with previous fit and add more estimator
            tol - Tolerance for stopping criteria
            C - Inverse of regularization strength
            max_iter - Maximum number of iterations taken for the solvers to converge

            solver - Algorithm to use in the optimization problem
            penalty - Specify the norm of the penalty
        '''
        self.model = LogisticRegression(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )

    def feature_importance(self):
        importances = self.model.coef_[0]

        feature_importances = pd.Series(importances, index=self.feature_names)

        fig, ax = plt.subplots()
        feature_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances of Logistic Regression")
        ax.set_ylabel("use of coefficients as importance scores")
        fig.tight_layout()
        plt.show()

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        solvers: list[str]=["newton-cg", "lbfgs", "liblinear"],
        penalty: list[str]=["l2"],
        c_values: list[int]=[100, 10, 1.0, 0.1, 0.01],
        scoring: str = "accuracy",
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        console_out: bool = False,
        train_afterwards: bool = False,
    ):
        '''
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            solver - Algorithm to use in the optimization problem
            penalty - Specify the norm of the penalty
            c_values - Inverse of regularization strength
            scoring - metrics to evaluate the models

            n_split_num - number of different splits
            n_repeats_num - number of repetition of one split

            console_out - output the the results of the different iterations
            train_afterwards - train the best model after finding it

        @return:
            set self.model = best model from search
        '''
        # define grid search
        grid = dict(solver=solvers, penalty=penalty, C=c_values)

        if console_out:
            print("grid: ", grid)

        cv = RepeatedStratifiedKFold(n_splits=n_split_num, n_repeats=n_repeats_num, random_state=42)
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=grid,
            n_jobs=-1,
            cv=cv,
            scoring=scoring,
            error_score=0,
        )

        logging.debug("starting hyperparameter tuning...")
        grid_result = grid_search.fit(x_train, y_train)
        logging.debug("... hyperparameter tuning finished")

        self.model = grid_result.best_estimator_

        if console_out:
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("mean: %f (stdev: %f) with: %r" % (mean, stdev, param))

        if train_afterwards:
            logging.debug("starting to train best model...")
            self.train(x_train, y_train)
            logging.debug("... best model trained")
