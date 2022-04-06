from typing import Union

import pandas as pd
from sklearn.neural_network import MLPClassifier

from .main_classifier import Classifier


class MLPC(Classifier):
    def __init__(
        self,
        model_name: str = "MLP Classifier",
        hidden_layer_sizes: tuple=(100,),
        activation: str="relu",
        solver: str="adam",
        alpha: float=0.0001,
        batch_size: Union[str, int]="auto",
        learning_rate: str="constant",
        learning_rate_init: float=0.001,
        power_t: float=0.5,
        max_iter: int=200,
        shuffle: bool=True,
        random_state: int=None,
        tol: float=0.0001,
        verbose: bool=False,
        warm_start: bool=False,
        momentum: float=0.9,
        nesterovs_momentum: bool=True,
        early_stopping: bool=False,
        validation_fraction: float=0.1,
        beta_1: float=0.9,
        beta_2: float=0.999,
        epsilon: float=1e-08,
        n_iter_no_change: int=10,
        max_fun: int=15000,
    ):
        """
        @param (important one):
            hidden_layer_sizes - the ith element represents the number of neurons in the ith hidden layer
            activation - activation function for the hidden layer
            solver - solver for weight optimization
            alpha - l2 penalty (regularization term) parameter
            learning_rate - learning rate schedule for weight updates
            warm_start - work with previous fit and add more estimator
            tol - Tolerance for stopping criteria
            max_iter - Maximum number of iterations taken for the solvers to converge

            random_state - random_state for model
            verbose - logging (True/False)
            batch_size - Size of minibatches for stochastic optimizers
            early_stopping - True: tests on 10% of train data and stops if there is for 'n_iter_no_change' no improvement in the metrics
        """
        self.model_name = model_name
        self.model_type = "MLPC"
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        hidden_layer_sizes: list[tuple] = [(10,30,10),(20,),(100,)],
        activation: list[str] = ['tanh', 'relu'],
        solver: list[str] = ['sgd', 'adam'],
        alpha: list[float] = [0.0001, 0.05],
        learning_rate: list[str] = ['constant','adaptive'],
        scoring: str = "accuracy",
        avg: str = "macro", 
        pos_label: Union[int,str] = 1,
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        verbose: int = 0,
        console_out: bool = False,
        train_afterwards: bool = False,
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            hidden_layer_sizes - the ith element represents the number of neurons in the ith hidden layer
            activation - activation function for the hidden layer
            solver - solver for weight optimization
            alpha - l2 penalty (regularization term) parameter
            learning_rate - learning rate schedule for weight updates

            scoring - metrics to evaluate the models
            avg - average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored

            n_split_num - number of different splits
            n_repeats_num - number of repetition of one split

            verbose - log level (higher number --> more logs)
            console_out - output the the results of the different iterations
            train_afterwards - train the best model after finding it

        @return:
            set self.model = best model from search
        """
        # define grid search
        grid = dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate)

        self.gridsearch(x_train, y_train, grid, scoring, n_split_num, n_repeats_num, verbose, console_out, train_afterwards)
