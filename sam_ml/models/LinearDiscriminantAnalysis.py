from typing import Union

import pandas as pd
from numpy import arange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .main_classifier import Classifier


class LDA(Classifier):
    def __init__(
        self,
        model_name: str = "LinearDiscriminantAnalysis",
        **kwargs,
    ):
        """
        @param (important one):
            solver - solver to use
            shrinkage - shrinkage parameters (does not work with 'svd' solver)
        """
        self.model_name = model_name
        self.model_type = "LDA"
        self.model = LinearDiscriminantAnalysis(**kwargs)

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        solver: list[str] = ["lsqr", "eigen"],
        shrinkage: list[Union[str, float]] = list(arange(0, 1, 0.01))+["auto"],
        rand_search: bool = True,
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

            solver - solver to use
            shrinkage - shrinkage parameters (does not work with 'svd' solver)

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
            solver=solver, shrinkage=shrinkage, **kwargs
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
