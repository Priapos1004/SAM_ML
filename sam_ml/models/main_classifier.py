import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from .main_model import Model


class Classifier(Model):
    def __init__(self, model_object=None, model_name="model"):
        self.model_name = model_name
        self.model_type = "Classifier"
        self.model = model_object

    def evaluate(
        self, x_test: pd.DataFrame, y_test: pd.Series, console_out: bool = True
    ):
        logging.debug("evaluation started...")
        pred = self.model.predict(x_test)

        if len(y_test.unique()) == 2:
            avg = None
        else:
            avg = "micro"

        # Calculate Accuracy, Precision and Recall Metrics
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average=avg)
        recall = recall_score(y_test, pred, average=avg)

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

    def feature_importance(self):
        if self.model_type == "MLPC":
            importances = [np.mean(i) for i in self.model.coefs_[0]] # MLP Classifier
        elif self.model_type == "DTC":
            importances = self.model.feature_importances_ # DecisionTree
        else:
            importances = self.model.coef_[0] # "normal"

        feature_importances = pd.Series(importances, index=self.feature_names)

        fig, ax = plt.subplots()
        feature_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances of "+self.model_name)
        ax.set_ylabel("use of coefficients as importance scores")
        fig.tight_layout()
        plt.show()

    def gridsearch(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        grid: dict,
        scoring: str = "accuracy",
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        verbose: int = 0,
        console_out: bool = False,
        train_afterwards: bool = True,
    ):
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
