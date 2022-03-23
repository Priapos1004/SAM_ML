from sklearn.svm import SVC as svc
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from matplotlib import pyplot as plt
from typing import Union
import logging
import pandas as pd
from .main_classifier import Classifier


class SVC(Classifier):
    def __init__(
        self,
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
            importances = self.model.coef_[0]

            feature_importances = pd.Series(importances, index=self.feature_names)

            fig, ax = plt.subplots()
            feature_importances.plot.bar(ax=ax)
            ax.set_title("Feature importances of SupportVectorClassifier")
            ax.set_ylabel("use of coefficients as importance scores")
            fig.tight_layout()
            plt.show()
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

        if console_out:
            print("grid: ", grid)

        cv = RepeatedStratifiedKFold(
            n_splits=n_split_num, n_repeats=n_repeats_num, random_state=42
        )
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=grid,
            n_jobs=-1,
            cv=cv,
            verbose=verbose,
            scoring=scoring,
            error_score=0,
        )

        logging.debug("starting hyperparameter tuning...")
        grid_result = grid_search.fit(x_train, y_train)
        logging.debug("... hyperparameter tuning finished")

        self.model = grid_result.best_estimator_
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        if console_out:
            means = grid_result.cv_results_["mean_test_score"]
            stds = grid_result.cv_results_["std_test_score"]
            params = grid_result.cv_results_["params"]
            print()
            for mean, stdev, param in zip(means, stds, params):
                print("mean: %f (stdev: %f) with: %r" % (mean, stdev, param))

        if train_afterwards:
            logging.debug("starting to train best model...")
            self.train(x_train, y_train)
            logging.debug("... best model trained")
