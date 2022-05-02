from typing import Union

import pandas as pd
from sklearn.svm import SVC as svc

from .main_classifier import Classifier


class SVC(Classifier):
    def __init__(
        self,
        model_name: str = "SupportVectorClassifier",
        kernel: str = "linear",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            random_state - random_state for model
            verbose - logging (True/False)
            C - Inverse of regularization strength

            kernel - kernel type to be used in the algorithm
            gamma - Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            c_values - Inverse of regularization strength
            max_iter - Maximum number of iterations taken for the solvers to converge

            cache_size - Specify the size of the kernel cache (in MB)
        """
        self.model_name = model_name
        self.model_type = "SVC"
        self.model = svc(
            kernel=kernel,
            random_state=random_state,
            **kwargs,
        )

    def feature_importance(self):
        if self.model.kernel == "linear":
            super(SVC, self).feature_importance()
        else:
            print(
                "feature importance is only available for a linear kernel. You are currently using: ",
                self.model.kernel,
            )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        kernel: list[str] = ["rbf"],
        gamma: list[Union[float, str]] = [1, 0.1, 0.01, 0.001, 0.0001, "scale", "auto"],
        C: list[int] = [0.1, 1, 10, 100, 1000],
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

            kernel - kernel type to be used in the algorithm
            gamma - Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
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
        grid = dict(kernel=kernel, gamma=gamma, C=C, **kwargs,)

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
