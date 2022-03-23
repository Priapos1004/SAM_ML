from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from matplotlib import pyplot as plt
import pandas as pd
import logging
from .main_classifier import Classifier
from typing import Union


class DTC(Classifier):
    def __init__(
        self,
        criterion: str="gini",
        splitter: str="best",
        max_depth: int=None,
        min_samples_split: Union[int, float]=2,
        min_samples_leaf: Union[int, float]=1,
        min_weight_fraction_leaf: float=0.0,
        max_features: Union[str, int, float]=None,
        random_state: int=None,
        max_leaf_nodes: int=None,
        min_impurity_decrease: float=0.0,
        class_weight: Union[dict, list[dict], str]=None,
        ccp_alpha: float=0.0,
    ):
        """
        @param (important one):
            criterion - function to measure the quality of a split
            max_depth - Maximum number of levels in tree
            min_samples_split - Minimum number of samples required to split a node
            min_samples_leaf - Minimum number of samples required at each leaf node
            random_state - random_state for model
        """
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
        )

    def feature_importance(self):
        importances = self.model.feature_importances_

        model_importances = pd.Series(importances, index=self.feature_names)

        fig, ax = plt.subplots()
        model_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances using MDI of DecisionTreeClassifier")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        criterion: list[str] = ["gini", "entropy"],
        max_depth: list[int] = range(1,10),
        min_samples_split: list[int] = range(2,10),
        min_samples_leaf: list[int] = range(1,5),
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        verbose: int = 1,
        scoring: str = "accuracy",
        console_out: bool = False,
        train_afterwards: bool = False,
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            criterion - function to measure the quality of a split
            max_depth - Maximum number of levels in tree
            min_samples_split - Minimum number of samples required to split a node
            min_samples_leaf - Minimum number of samples required at each leaf node

            n_split_num - number of different splits
            n_repeats_num - number of repetition of one split

            verbose - log level (higher number --> more logs)
            scoring - metrics to evaluate the models
            console_out - output the the results of the different iterations
            train_afterwards - train the best model after finding it

        @return:
            set self.model = best model from search
        """
        # Create the random grid
        grid = dict(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)

        if console_out:
            print("grid: ", grid)

        cv = RepeatedStratifiedKFold(n_splits=n_split_num, n_repeats=n_repeats_num, random_state=42)

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
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            print()
            for mean, stdev, param in zip(means, stds, params):
                print("mean: %f (stdev: %f) with: %r" % (mean, stdev, param))

        if train_afterwards:
            logging.debug("starting to train best model...")
            self.train(x_train, y_train)
            logging.debug("... best model trained")
