from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sam_ml.models.main_classifier import classifier
from typing import Union


class rfc(classifier):
    def __init__(
        self,
        n_estimators: int=100,
        criterion="gini", # “gini” or “entropy”
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap: bool=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
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
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()

    def hyperparameter_tuning(self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: list[int] =[int(x) for x in range(200, 2000, 200)],
        max_features: list[str]=["auto", "sqrt"],
        max_depth: list[int]=[int(x) for x in np.linspace(10, 110, num=11)] + [None],
        min_samples_split: list[int]=[2, 5, 10],
        min_samples_leaf: list[int]=[1, 2, 4],
        bootstrap: list[bool]=[True, False],
        n_iter_num: int = 75,
        cv_num: int = 3,
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

            Random search of parameters, using "cv_num" fold cross validation,
            search across "n_iter_num" different combinations, and use all available cores

        @return:
            set self.model = best model from search
        '''
        # Create the random grid
        self.random_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
        }

        # random search
        rf_random = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.random_grid,
            n_iter=n_iter_num,
            cv=cv_num,
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )
        # Fit the random search model
        rf_random.fit(x_train, y_train)

        print("rf_random.best_params_:")
        print(rf_random.best_params_)

        print("rf_random.best_estimator_:")
        print(rf_random.best_estimator_)
        self.model = rf_random.best_estimator_
