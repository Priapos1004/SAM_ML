from typing import Union

import pandas as pd
from numpy import arange
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .main_classifier import Classifier


class BC(Classifier):
    def __init__(
        self,
        model_name: str = "BaggingClassifier",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        @param (important one):
            base_estimator -base estimator from which the boosted ensemble is built (default: DecisionTreeClassifier with max_depth=1)
            n_estimator - number of boosting stages to perform
            max_samples - the number of samples to draw from X to train each base estimator
            max_features - the number of features to draw from X to train each base estimator
            bootstrap - whether samples are drawn with replacement. If False, sampling without replacement is performed
            bootstrap_features - whether features are drawn with replacement
        """
        self.model_name = model_name
        self.model_type = "BC"
        self.model = BaggingClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        base_estimator = [DecisionTreeClassifier(max_depth=i) for i in range(1,11)]+[SVC(probability=True, kernel='linear'), LogisticRegression(), GradientBoostingClassifier(), RandomForestClassifier(max_depth=5), KNeighborsClassifier()],
        n_estimators: list[int] = list(range(10, 101, 10)) + [3, 4, 5, 6, 7, 8, 9, 200, 500, 1000, 1500, 3000],
        max_samples: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        max_features: list[int] = [0.5, 1, 2, 4],
        bootstrap: list[bool] = [True, False],
        bootstrap_features: list[bool] = [True, False],
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
            max_samples - the number of samples to draw from X to train each base estimator
            max_features - the number of features to draw from X to train each base estimator
            bootstrap - whether samples are drawn with replacement. If False, sampling without replacement is performed
            bootstrap_features - whether features are drawn with replacement

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
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
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
