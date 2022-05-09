from typing import Union

import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier

from .main_classifier import Classifier


class GPC(Classifier):
    def __init__(
        self,
        model_name: str = "GaussianProcessClassifier",
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            multi_class - specifies how multi-class classification problems are handled
            max_iter_predict - the maximum number of iterations in Newton's method for approximating the posterior during predict
        """
        self.model_name = model_name
        self.model_type = "GPC"
        self.model = GaussianProcessClassifier(
            n_jobs=n_jobs, random_state=random_state, **kwargs,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        multi_class: list[str] = ["one_vs_rest", "one_vs_one"],
        max_iter_predict: list[int] = [1, 10, 50, 100, 200, 500, 1000],
        rand_search: bool = False,
        n_iter_num: int = 75,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        verbose: int = 0,
        console_out: bool = False,
        train_afterwards: bool = True,
        **kwargs,
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            multi_class - specifies how multi-class classification problems are handled
            max_iter_predict - the maximum number of iterations in Newton's method for approximating the posterior during predict

            rand_search - True: RandomizedSearchCV, False: GridSearchCV
            n_iter_num - Combinations to try out if rand_search=True

            n_split_num - number of different splits
            n_repeats_num - number of repetition of one split
            
            scoring - metrics to evaluate the models
            avg - average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored
            
            verbose - log level (higher number --> more logs)
            console_out - output the the results of the different iterations
            train_afterwards - train the best model after finding it

        @return:
            set self.model = best model from search
        """
        # Create the random grid
        grid = dict(
            multi_class=multi_class, max_iter_predict=max_iter_predict, **kwargs
        )

        self.gridsearch(
            x_train=x_train,
            y_train=y_train,
            grid=grid,
            scoring=scoring,
            avg=avg,
            pos_label=pos_label,
            rand_search=rand_search,
            n_iter_num=n_iter_num,
            n_split_num=n_split_num,
            n_repeats_num=n_repeats_num,
            verbose=verbose,
            console_out=console_out,
            train_afterwards=train_afterwards,
        )