from typing import Union

import pandas as pd
from sklearn.naive_bayes import BernoulliNB

from .main_classifier import Classifier


class BNB(Classifier):
    def __init__(
        self,
        model_name: str = "BernoulliNB",
        **kwargs,
    ):
        """
        @params (important one):
            binarize: threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors
            fit_prior: whether to learn class prior probabilities or not. If false, a uniform prior will be used
        """
        self.model_name = model_name
        self.model_type = "BNB"
        self.model = BernoulliNB(**kwargs,)

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        fit_prior: list[bool] = [True, False],
        binarize: list[float] = list(range(0, 10)),
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
            x_train: DataFrame with train features
            y_train: Series with labels

            binarize: threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors
            fit_prior: whether to learn class prior probabilities or not. If false, a uniform prior will be used

            scoring: metrics to evaluate the models
            avg: average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label: if avg="binary", pos_label says which class to score. Else pos_label is ignored

            rand_search: True: RandomizedSearchCV, False: GridSearchCV
            n_iter_num: Combinations to try out if rand_search=True

            n_split_num: number of different splits
            n_repeats_num: number of repetition of one split

            verbose: log level (higher number --> more logs)
            console_out: output the the results of the different iterations
            train_afterwards: train the best model after finding it

        @return:
            set self.model = best model from search
        """
        # define grid search
        grid = dict(fit_prior=fit_prior, binarize=binarize, **kwargs,)

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
