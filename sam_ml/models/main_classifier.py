import logging
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, cross_validate)

from .main_model import Model


class Classifier(Model):
    def __init__(self, model_object=None, model_name="Classifier"):
        self.model_name = model_name
        self.model_type = "Classifier"
        self.model = model_object
        self.cv_scores = {}

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

    def cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 3,
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        return_estimator: bool = False,
        console_out: bool = True,
        return_as_dict: bool = False,
    ) -> Union[dict[list], pd.DataFrame]:
        """
        @param:
            X, y - data to cross validate on
            cv_num - number of different splits

            avg - average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored

            return_estimator - if the estimator from the different splits shall be returned (suggestion: return_as_dict = True)

            console_out - shall the result be printed into the console
            return_as_dict - True: return scores as a dict, False: return scores as a pandas DataFrame

        @return:
            depending on "return_as_dict"
            the scores will be saved in self.cv_scores as dict (WARNING: return_estimator=True increases object size)
        """

        precision_scorer = make_scorer(precision_score, average=avg, pos_label=pos_label)
        recall_scorer = make_scorer(recall_score, average=avg, pos_label=pos_label)

        if avg == "binary":
            scorer = {
                f"precision ({avg}, label={pos_label})": precision_scorer,
                f"recall ({avg}, label={pos_label})": recall_scorer,
                "accuracy": "accuracy",
            }
        else:
            scorer = {
                f"precision ({avg})": precision_scorer,
                f"recall ({avg})": recall_scorer,
                "accuracy": "accuracy",
            }

        logging.debug("starting to cross validate...")
        cv_scores = cross_validate(
            self.model,
            X,
            y,
            scoring=scorer,
            cv=cv_num,
            return_train_score=True,
            return_estimator=return_estimator,
            n_jobs=-1,
        )
        logging.debug("... cross validation completed")

        pd_scores = pd.DataFrame(cv_scores).transpose()
        pd_scores["average"] = pd_scores.mean(numeric_only=True, axis=1)

        self.cv_scores = pd_scores.to_dict()

        if console_out:
            print(pd_scores)

        if return_as_dict:
            return self.cv_scores
        else:
            return pd_scores

    def feature_importance(self) -> plt.show:
        """
        feature_importance() generates a matplotlib plot of the feature importance from self.model
        """
        if self.model_type == "MLPC":
            importances = [np.mean(i) for i in self.model.coefs_[0]]  # MLP Classifier
        elif self.model_type in ["DTC","RFC","GBM"]:
            importances = self.model.feature_importances_  # DecisionTree
        else:
            importances = self.model.coef_[0]  # "normal"

        feature_importances = pd.Series(importances, index=self.feature_names)

        fig, ax = plt.subplots()
        if self.model_type in ["RFC", "GBM"]:
            if self.model_type == "RFC":
                std = np.std(
                    [tree.feature_importances_ for tree in self.model.estimators_], axis=0,
                )
            elif self.model_type == "GBM":
                std = np.std(
                    [tree[0].feature_importances_ for tree in self.model.estimators_], axis=0,
                )
            feature_importances.plot.bar(yerr=std, ax=ax)
        else:
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
            print()

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
                scoring=scoring,
                error_score=0,
            )

        logging.debug("starting hyperparameter tuning...")
        grid_result = grid_search.fit(x_train, y_train)
        logging.debug("... hyperparameter tuning finished")

        if console_out:
            means = grid_result.cv_results_["mean_test_score"]
            stds = grid_result.cv_results_["std_test_score"]
            params = grid_result.cv_results_["params"]
            print()
            for mean, stdev, param in zip(means, stds, params):
                print("mean: %f (stdev: %f) with: %r" % (mean, stdev, param))

        self.model = grid_result.best_estimator_
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        if train_afterwards:
            logging.debug("starting to train best model...")
            self.train(x_train, y_train, console_out=console_out)
            logging.debug("... best model trained")
