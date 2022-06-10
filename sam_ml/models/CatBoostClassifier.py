from typing import Union

import pandas as pd
from catboost import CatBoostClassifier

from .main_classifier import Classifier


class CBC(Classifier):
    def __init__(
        self,
        model_name: str = "CatBoostClassifier",
        verbose: Union[bool, int] = False,
        random_state: int = 42,
        allow_writing_files: bool = False,
        **kwargs,
    ):
        """
        @info:
            no type hints due to documentation of CatBoost library
        @param (important one):
            depth - depth of the tree
            learning_rate - learning rate
            iterations - maximum number of trees that can be built when solving machine learning problems
            bagging_temperature - defines the settings of the Bayesian bootstrap
            random_strength - the amount of randomness to use for scoring splits when the tree structure is selected
            l2_leaf_reg - coefficient at the L2 regularization term of the cost function
            border_count - the number of splits for numerical features
        """
        self.model_name = model_name
        self.model_type = "CBC"
        self.model = CatBoostClassifier(
            verbose=verbose,
            random_state=random_state,
            allow_writing_files=allow_writing_files,
            **kwargs,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        depth: list[int] = [4, 5, 6, 7, 8, 9, 10],
        learning_rate: list[float] = [0.1, 0.01, 0.02, 0.03, 0.04],
        iterations: list[int] = list(range(10, 101, 10)) + [200, 500, 1000],
        bagging_temperature: list[float] = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0],
        random_strength: list[float] = [
            0.000000001,
            0.0000001,
            0.00001,
            0.001,
            0.1,
            1,
            10,
        ],
        l2_leaf_reg: list[int] = [2, 4, 6, 8, 12, 16, 20, 24, 30],
        border_count: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 254],
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        verbose: int = 0,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        rand_search: bool = True,
        n_iter_num: int = 75,
        console_out: bool = False,
        train_afterwards: bool = True,
        **kwargs,
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            depth - depth of the tree
            learning_rate - learning rate
            iterations - maximum number of trees that can be built when solving machine learning problems
            bagging_temperature - defines the settings of the Bayesian bootstrap
            random_strength - the amount of randomness to use for scoring splits when the tree structure is selected
            l2_leaf_reg - coefficient at the L2 regularization term of the cost function
            border_count - the number of splits for numerical features

            n_split_num - number of different splits
            n_repeats_num - number of repetition of one split

            scoring - metrics to evaluate the models
            avg - average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored
            rand_search - True: RandomizedSearchCV, False: GridSearchCV
            n_iter_num - Combinations to try out if rand_search=True

            verbose - log level (higher number --> more logs)
            console_out - output the the results of the different iterations
            train_afterwards - train the best model after finding it

        @return:
            set self.model = best model from search
        """
        # Create the random grid
        grid = dict(
            depth=depth,
            learning_rate=learning_rate,
            iterations=iterations,
            bagging_temperature=bagging_temperature,
            random_strength=random_strength,
            l2_leaf_reg=l2_leaf_reg,
            border_count=border_count,
            **kwargs,
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
