from typing import Union

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from .main_classifier import Classifier


class GBM(Classifier):
    def __init__(
        self,
        model_name: str = "GradientBoostingMachine",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            n_estimator - number of boosting stages to perform
            criterion - function to measure the quality of a split
            max_depth - Maximum number of levels in tree
            min_samples_split - Minimum number of samples required to split a node
            min_samples_leaf - Minimum number of samples required at each leaf node
            max_features - number of features to consider when looking for the best split
            subsample - fraction of samples to be used for fitting the individual base learners
            loss - The loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss 'exponential' gradient boosting recovers the AdaBoost algorithm
            learning_rate - shrinks the contribution of each tree by learning rate

            warm_start - work with previous fit and add more estimator
            random_state - random_state for model
        """
        self.model_name = model_name
        self.model_type = "GBM"
        self.model = GradientBoostingClassifier(
            random_state=random_state,
            **kwargs,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: list[int] = list(range(20,101,10))+[200, 500, 1000, 1500],
        max_depth: list[int] = list(range(1,8))+[10,12,15],
        min_samples_split: list[Union[int, float]] = [2,4,6,8,10,20,40,60,100],
        min_samples_leaf: list[Union[int, float]] = [2,4,6,8,10,20,40,60,100],
        max_features: list[Union[str, int, float]] = ['auto', 'sqrt', 'log2', None],
        subsample: list[float] = [0.7,0.75,0.8,0.85,0.9,0.95,1],
        criterion: list[str] = ['friedman_mse', 'mse'],
        loss: list[str] = ["deviance", "exponential"],
        learning_rate: list[float] = [0.1, 0.05, 0.01, 0.005],
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
        **kwargs,
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            n_estimator - number of boosting stages to perform
            criterion - function to measure the quality of a split
            max_depth - Maximum number of levels in tree
            min_samples_split - Minimum number of samples required to split a node
            min_samples_leaf - Minimum number of samples required at each leaf node
            max_features - number of features to consider when looking for the best split
            subsample - fraction of samples to be used for fitting the individual base learners
            loss - The loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss 'exponential' gradient boosting recovers the AdaBoost algorithm
            learning_rate - shrinks the contribution of each tree by learning rate

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
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            subsample=subsample,
            loss=loss,
            learning_rate=learning_rate,
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
