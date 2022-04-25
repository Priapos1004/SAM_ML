from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from .main_classifier import Classifier


class RFC(Classifier):
    def __init__(
        self,
        model_name: str = "RandomForestClassifier",
        n_estimators: int = 100,
        criterion: str = "gini",  # “gini” or “entropy”
        max_depth: int = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[str, int, float] = "auto",
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = -1,  # how many cores shall be used
        random_state: int = None,
        verbose: int = 0,
        warm_start: bool = False,  # True --> work wih the previous fit and add more estimators
        class_weight: Union[dict, list[dict]] = None,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float] = None,
    ):
        """
        @param (important one):
            n_estimators - Number of trees in random forest
            max_depth - Maximum number of levels in tree
            n_jobs - how many cores shall be used (-1 means all)
            random_state - random_state for model
            verbose - log level (higher number --> more logs)
            warm_start - work with previous fit and add more estimator

            max_features - Number of features to consider at every split
            min_samples_split - Minimum number of samples required to split a node
            min_samples_leaf - Minimum number of samples required at each leaf node
            bootstrap - Method of selecting samples for training each tree
        """
        self.model_name = model_name
        self.model_type = "RFC"
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: list[int] = [1, 2, 4, 8, 16, 32, 64, 100, 200, 500, 1000],
        max_features: list[Union[str, int, float]] = ["auto", "sqrt"],
        max_depth: list[int] = [5,6,7,8,10,15],
        min_samples_split: list[int] = [2, 3, 5, 10],
        min_samples_leaf: list[int] = [1, 2, 4],
        bootstrap: list[bool] = [True, False],
        criterion: list[str] = ["gini", "entropy"],
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
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            n_estimators - Number of trees in random forest
            max_features - Number of features to consider at every split
            max_depth - Maximum number of levels in tree
            min_samples_split - Minimum number of samples required to split a node
            min_samples_leaf - Minimum number of samples required at each leaf node
            bootstrap - Method of selecting samples for training each tree
            criterion - function to measure the quality of a split

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
        grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
            "criterion": criterion,
        }

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
