from typing import Union

import pandas as pd
from sklearn.svm import LinearSVC

from .main_classifier import Classifier


class LSVC(Classifier):
    def __init__(
        self,
        model_name: str = "LinearSupportVectorClassifier",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            random_state - random_state for model
            verbose - logging
            penalty - specifies the norm used in the penalization
            dual - select the algorithm to either solve the dual or primal optimization problem
            C - Inverse of regularization strength
            max_iter - Maximum number of iterations taken for the solvers to converge
        """
        self.model_name = model_name
        self.model_type = "LSVC"
        self.model = LinearSVC(
            random_state=random_state,
            **kwargs,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        penalty: list[str] = ["l1","l2"],
        dual: list[bool] = [True, False],
        C: list[float] = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000, 100000],
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        rand_search: bool = True,
        n_iter_num: int = 75,
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        verbose: int = 1,
        console_out: bool = False,
        train_afterwards: bool = True,
        **kwargs,
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            penalty - specifies the norm used in the penalization
            dual - select the algorithm to either solve the dual or primal optimization problem
            C - Inverse of regularization strength

            scoring - metrics to evaluate the models
            avg - average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored

            rand_search - True: RandomizedSearchCV, False: GridSearchCV
            n_iter_num - Combinations to try out if rand_search=True

            n_split_num - number of different splits
            n_repeats_num - number of repetition of one split

            verbose - log level (higher number --> more logs)
            console_out - output the the results of the different iterations
            train_afterwards - train the best model after finding it

        @return:
            set self.model = best model from search
        """
        # define grid search
        grid = dict(penalty=penalty, dual=dual, C=C, **kwargs,)

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
