from typing import Union

import pandas as pd
from sklearn.neural_network import MLPClassifier

from .main_classifier import Classifier


class MLPC(Classifier):
    def __init__(
        self,
        model_name: str = "MLP Classifier",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            hidden_layer_sizes: the ith element represents the number of neurons in the ith hidden layer
            activation: activation function for the hidden layer
            solver: solver for weight optimization
            alpha: l2 penalty (regularization term) parameter
            learning_rate: learning rate schedule for weight updates
            warm_start: work with previous fit and add more estimator
            tol: Tolerance for stopping criteria
            max_iter: Maximum number of iterations taken for the solvers to converge

            random_state: random_state for model
            verbose: logging (True/False)
            batch_size: Size of minibatches for stochastic optimizers
            early_stopping: True: tests on 10% of train data and stops if there is for 'n_iter_no_change' no improvement in the metrics
        """
        self.model_name = model_name
        self.model_type = "MLPC"
        self.model = MLPClassifier(
            random_state=random_state,
            **kwargs,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        hidden_layer_sizes: list[tuple] = [(10, 30, 10), (20,), (10,), (100,), (50,50,50), (50,100,50)],
        activation: list[str] = ["tanh", "relu", "logistic"],
        solver: list[str] = ["sgd", "adam"],
        alpha: list[float] = [0.0001, 0.001, 0.01, 0.05],
        learning_rate: list[str] = ["constant", "adaptive"],
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        rand_search: bool = True,
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

            hidden_layer_sizes: the ith element represents the number of neurons in the ith hidden layer
            activation: activation function for the hidden layer
            solver: solver for weight optimization
            alpha: l2 penalty (regularization term) parameter
            learning_rate: learning rate schedule for weight updates

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
        grid = dict(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
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
