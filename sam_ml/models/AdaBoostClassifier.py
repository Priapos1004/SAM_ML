from typing import Union

import pandas as pd
from numpy import arange
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .main_classifier import Classifier


class ABC(Classifier):
    def __init__(
        self,
        model_name: str = "AdaBoostClassifier",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            base_estimator -base estimator from which the boosted ensemble is built (default: DecisionTreeClassifier with max_depth=1)
            n_estimator - number of boosting stages to perform
            learning_rate - shrinks the contribution of each tree by learning rate
            algorithm - boosting algorithm
            random_state - random_state for model
        """
        self.model_name = model_name
        self.model_type = "ABC"
        self.model = AdaBoostClassifier(
            random_state=random_state,
            **kwargs,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        base_estimator = [DecisionTreeClassifier(max_depth=i) for i in range(1,11)]+[SVC(probability=True, kernel='linear'), LogisticRegression(), GradientBoostingClassifier(), RandomForestClassifier(max_depth=5)],
        n_estimators: list[int] = list(range(10, 101, 10)) + [200, 500, 1000, 1500, 3000],
        learning_rate: list[float] = [0.1, 0.05, 0.01, 0.005]+list(arange(0.2,2.1,0.1)),
        algorithm: list[str] = ["SAMME.R", "SAMME"],
        n_split_num: int = 5,
        n_repeats_num: int = 2,
        verbose: int = 1,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        rand_search: bool = True,
        n_iter_num: int = 100,
        console_out: bool = False,
        train_afterwards: bool = True,
        **kwargs,
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            base_estimator -base estimator from which the boosted ensemble is built (default: DecisionTreeClassifier with max_depth=1)
            n_estimator - number of boosting stages to perform
            learning_rate - shrinks the contribution of each tree by learning rate
            algorithm - boosting algorithm

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
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
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
