from typing import Union

import pandas as pd
from sklearn.linear_model import LogisticRegression

from .main_classifier import Classifier


class LR(Classifier):
    def __init__(
        self,
        model_name: str = "LogisticRegression",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            n_jobs - how many cores shall be used (-1 means all) (n_jobs > 1 does not have any effect when 'solver' is set to 'liblinear)
            random_state - random_state for model
            verbose - log level (higher number --> more logs)
            warm_start - work with previous fit and add more estimator
            tol - Tolerance for stopping criteria
            C - Inverse of regularization strength
            max_iter - Maximum number of iterations taken for the solvers to converge

            solver - Algorithm to use in the optimization problem
            penalty - Specify the norm of the penalty
        """
        self.model_name = model_name
        self.model_type = "LR"
        self.model = LogisticRegression(
            random_state=random_state,
            **kwargs,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        solvers: list[str] = ["newton-cg", "lbfgs", "liblinear", "sag"],
        penalty: list[str] = ["l2"],
        C: list[int] = [100, 10, 1.0, 0.1, 0.01],
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        rand_search: bool = False,
        n_iter_num: int = 75,
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

            solver - Algorithm to use in the optimization problem
            penalty - Specify the norm of the penalty
            c_values - Inverse of regularization strength

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
        grid = dict(solver=solvers, penalty=penalty, C=C, **kwargs,)

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
