import logging
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     RepeatedStratifiedKFold)

from .main_model import Model


class Classifier(Model):
    def __init__(self, model_object=None, model_name="Classifier"):
        self.model_name = model_name
        self.model_type = "Classifier"
        self.model = model_object

    def evaluate(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        avg: str = None,
        pos_label: Union[int, str] = 1,
        console_out: bool = True,
    ) -> dict:
        """
        @param:
            x_test, y_test - Data to evaluate model
            avg - average to use for precision and recall score (e.g.: "micro", None, "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored
            console_out - shall the result be printed into the console
        """
        logging.debug("evaluation started...")
        pred = self.model.predict(x_test)

        # Calculate Accuracy, Precision and Recall Metrics
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average=avg, pos_label=pos_label)
        recall = recall_score(y_test, pred, average=avg, pos_label=pos_label)

        if console_out:
            print("accuracy: ", accuracy)
            print("precision: ", precision)
            print("recall: ", recall)

            print("classification report: ")
            print(classification_report(y_test, pred))

        score = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

        logging.debug("... evaluation finished")
        return score

    def feature_importance(self) -> plt.show:
        '''
        feature_importance() generates a matplotlib plot of the feature importance from self.model
        '''
        if self.model_type == "MLPC":
            importances = [np.mean(i) for i in self.model.coefs_[0]]  # MLP Classifier
        elif self.model_type == "DTC":
            importances = self.model.feature_importances_  # DecisionTree
        else:
            importances = self.model.coef_[0]  # "normal"

        feature_importances = pd.Series(importances, index=self.feature_names)

        fig, ax = plt.subplots()
        feature_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances of " + self.model_name)
        ax.set_ylabel("use of coefficients as importance scores")
        fig.tight_layout()
        plt.show()

    def gridsearch(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        grid: dict,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        verbose: int = 0,
        rand_search: bool = True,
        n_iter_num: int = 75,
        console_out: bool = False,
        train_afterwards: bool = True,
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            grid - dictonary of parameters to tune

            scoring - metrics to evaluate the models
            avg - average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored

            rand_search - True: RandomizedSearchCV, False: GridSearchCV
            n_iter_num - Combinations to try out if rand_search=True

            n_split_num - number of different splits
            n_repeats_num - number of repetition of one split

            verbose - log level (higher number --> more logs)
            console_out - output the the results of the different iterations
            train_afterwards - train the best model after finding it

        @return:
            set self.model = best model from search
        """
        if console_out:
            print("grid: ", grid)

        if scoring == "precision":
            scoring = make_scorer(precision_score, average=avg, pos_label=pos_label)
        elif scoring == "recall":
            scoring = make_scorer(recall_score, average=avg, pos_label=pos_label)

        cv = RepeatedStratifiedKFold(
            n_splits=n_split_num, n_repeats=n_repeats_num, random_state=42
        )

        if rand_search:
            grid_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=grid,
                n_iter=n_iter_num,
                cv=cv,
                verbose=verbose,
                random_state=42,
                n_jobs=-1,
                scoring=scoring,
            )
        else:
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=grid,
                n_jobs=-1,
                cv=cv,
                verbose=verbose,
                random_state=42,
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
            self.train(x_train, y_train, console_out=False)
            logging.debug("... best model trained")
