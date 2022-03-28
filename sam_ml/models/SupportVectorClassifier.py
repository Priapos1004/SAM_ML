from sklearn.svm import SVC as svc
from typing import Union
import pandas as pd
from .main_classifier import Classifier


class SVC(Classifier):
    def __init__(
        self,
        model_name: str = "SupportVectorClassifier",
        C: float=1.0,
        kernel: str="linear",
        degree: int=3,
        gamma: Union[str, float]="scale",
        coef0: float=0.0,
        shrinking: bool=True,
        probability: bool=False,
        tol: float=0.001,
        cache_size: float=200,
        class_weight: Union[dict, str]=None,
        verbose: bool=False,
        max_iter: int=-1,
        decision_function_shape: str="ovr",
        break_ties: float=False,
        random_state: int=None,
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
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

    def feature_importance(self):
        if self.model.kernel == "linear":
            super(SVC, self).feature_importance()
        else:
            print("feature importance is only available for a linear kernel. You are currently using: ", self.model.kernel)

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        kernel: list[str] = ["rbf"],
        gamma: list[Union[float, str]] = [1, 0.1, 0.01, 0.001, 0.0001, "scale", "auto"],
        c_values: list[int] = [0.1, 1, 10, 100, 1000],
        scoring: str = "accuracy",
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        verbose: int = 1,
        console_out: bool = False,
        train_afterwards: bool = False,
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            kernel - kernel type to be used in the algorithm
            gamma - Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            c_values - Inverse of regularization strength

            scoring - metrics to evaluate the models

            n_split_num - number of different splits
            n_repeats_num - number of repetition of one split

            verbose - log level (higher number --> more logs)
            console_out - output the the results of the different iterations
            train_afterwards - train the best model after finding it

        @return:
            set self.model = best model from search
        """
        # define grid search
        grid = dict(kernel=kernel, gamma=gamma, C=c_values)

        self.gridsearch(x_train, y_train, grid, scoring, n_split_num, n_repeats_num, verbose, console_out, train_afterwards)
