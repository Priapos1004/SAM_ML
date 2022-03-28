from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from .main_classifier import Classifier
from typing import Union


class DTC(Classifier):
    def __init__(
        self,
        model_name: str = "DecisionTreeClassifier",
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
        self.model_name = model_name
        self.model_type = "DTC"
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

        self.gridsearch(x_train, y_train, grid, scoring, n_split_num, n_repeats_num, verbose, console_out, train_afterwards)
