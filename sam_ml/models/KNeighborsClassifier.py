from typing import Union

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from .main_classifier import Classifier


class KNC(Classifier):
    def __init__(
        self,
        model_name: str = "KNeighborsClassifier",
        n_neighbors: int = 5,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        metric_params: dict = None,
        n_jobs: int = None,
    ):
        """
        @param (important one):
            n_neighbors - Number of neighbors to use by default for kneighbors queries
            weights - Weight function used in prediction
            algorithm - Algorithm used to compute the nearest neighbors
            leaf_size - Leaf size passed to BallTree or KDTree
            p - number of metric that is used (manhattan, euclidean, minkowski)
            n_jobs - the number of parallel jobs to run for neighbors search [problem with n_jobs = -1 --> kernel dies]
        """
        self.model_name = model_name
        self.model_type = "KNC"
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_neighbors: list[int] = list(range(1,30)),
        p: list[int] = [1,2,3,4,5],
        leaf_size: list[int] = list(range(1,50)),
        weights: list[str] = ["uniform", "distance"],
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        verbose: int = 1,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        rand_search: bool = True,
        n_iter_num: int = 75,
        console_out: bool = False,
        train_afterwards: bool = True,
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            n_neighbors - Number of neighbors to use by default for kneighbors queries
            p - number of metric that is used (manhattan, euclidean, minkowski)
            leaf_size - Leaf size passed to BallTree or KDTree

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
            n_neighbors=n_neighbors,
            p=p,
            leaf_size=leaf_size,
            weights=weights,
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
