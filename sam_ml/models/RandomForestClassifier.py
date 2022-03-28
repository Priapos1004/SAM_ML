import logging
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from .main_classifier import Classifier


class RFC(Classifier):
    def __init__(
        self,
        model_name: str = "RandomForestClassifier",
        n_estimators: int=100,
        criterion: str="gini", # “gini” or “entropy”
        max_depth: int=None,
        min_samples_split: Union[int, float]=2,
        min_samples_leaf: Union[int, float]=1,
        min_weight_fraction_leaf: float=0.0,
        max_features: Union[str, int, float]="auto",
        max_leaf_nodes: int=None,
        min_impurity_decrease: float=0.0,
        bootstrap: bool=True,
        oob_score: bool=False,
        n_jobs: int=-1, # how many cores shall be used
        random_state: int=None,
        verbose: int=0,
        warm_start: bool=False, # True --> work wih the previous fit and add more estimators
        class_weight: Union[dict, list[dict]]=None,
        ccp_alpha: float=0.0,
        max_samples: Union[int, float]=None,
    ):
        '''
        @param (important one):
            n_estimators - Number of trees in random forest
            max_depth - Maximum number of levels in tree
            n_jobs - how many cores shall be used (-1 means all)
            random_state - random_state for model
            verbose - log level (higher number --> more logs)
            warm_start - work with previous fit and add more estimator

            max_features - Number of features to consider at every split
            min_samples_split - Minimum number of samples required to split a node
            min_samples_leaf - Minimum number of samples required at each leaf node
            bootstrap - Method of selecting samples for training each tree
        '''
        self.model_name = model_name
        self.model_type = "RFC"
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def feature_importance(self):
        importances = self.model.feature_importances_

        std = np.std(
            [tree.feature_importances_ for tree in self.model.estimators_],
            axis=0,
        )
        forest_importances = pd.Series(importances, index=self.feature_names)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI of RandomForestClassifier")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()

    def hyperparameter_tuning(self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: list[int] =[int(x) for x in range(200, 2000, 200)],
        max_features: list[Union[str, int, float]]=["auto", "sqrt"],
        max_depth: list[int]=[int(x) for x in np.linspace(10, 110, num=11)] + [None],
        min_samples_split: list[int]=[2, 5, 10],
        min_samples_leaf: list[int]=[1, 2, 4],
        bootstrap: list[bool]=[True, False],
        criterion: list[str] = ["gini", "entropy"],
        n_iter_num: int = 75,
        cv_num: int = 3,
        scoring:str = "accuracy",
        console_out: bool = False,
        train_afterwards: bool = False
        ):
        '''
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            n_estimators - Number of trees in random forest
            max_features - Number of features to consider at every split
            max_depth - Maximum number of levels in tree
            min_samples_split - Minimum number of samples required to split a node
            min_samples_leaf - Minimum number of samples required at each leaf node
            bootstrap - Method of selecting samples for training each tree
            criterion - function to measure the quality of a split

            Random search of parameters, using "cv_num" fold cross validation,
            search across "n_iter_num" different combinations, and use all available cores

            scoring - metrics to evaluate the models
            console_out - output the the results of the different iterations
            train_afterwards - train the best model after finding it

        @return:
            set self.model = best model from search
        '''
        # Create the random grid
        random_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
            "criterion": criterion,
        }

        if console_out:
            print("grid: ", random_grid)

        # random search
        rf_random = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=random_grid,
            n_iter=n_iter_num,
            cv=cv_num,
            verbose=2,
            random_state=42,
            n_jobs=-1,
            scoring=scoring
        )

        logging.debug("starting hyperparameter tuning...")
        # Fit the random search model
        rf_random.fit(x_train, y_train)

        logging.debug("... finished hyperparameter tuning")

        if console_out:
            print("rf_random.best_params_:")
            print(rf_random.best_params_)

        print("rf_random.best_estimator_:")
        print(rf_random.best_estimator_)

        logging.debug("set self.model to best estimator")
        self.model = rf_random.best_estimator_

        if train_afterwards:
            logging.debug("starting to train best estimator...")
            self.train(x_train, y_train)
            logging.debug("... best estimator trained")
