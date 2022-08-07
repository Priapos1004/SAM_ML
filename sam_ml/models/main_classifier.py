import logging
from datetime import timedelta
from statistics import mean
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, cross_validate)
from tqdm.auto import tqdm

from sam_ml.data.embeddings import Embeddings_builder
from sam_ml.data.sampling import Sampler
from sam_ml.data.scaler import Scaler

from .main_model import Model
from .scorer import l_scoring, s_scoring


class Classifier(Model):
    def __init__(self, model_object = None, model_name: str = "classifier", model_type: str = "Classifier", grid: dict[str, list] = {}, is_pipeline: bool = False):
        """
        @params:
            model_object: model with 'fit' and 'predict' method
            model_name: name of the model
            model_type: kind of estimator
            grid: hyperparameter grid for the model
            is_pipeline: is the model a sklearn pipeline
        """
        super().__init__(model_object, model_name, model_type)
        self._grid = grid
        self.is_pipeline = is_pipeline
        self.cv_scores: dict[str, float] = {}

    @property
    def grid(self):
        return self._grid

    def update_grid(self, **kwargs):
        """
        function to update self.grid 

        e.g.:
            - model.grid {"n_estimators": [3, 4, 5]}
            - model.update_grid(n_estimators = [10, 3, 5], solver = ["sag", "l1"])
            - model.grid {"n_estimators": [10, 3, 5], "solver": ["sag", "l1"]}
        """
        for param in list(dict(**kwargs).keys()):
            self._grid[param] = dict(**kwargs)[param]

    def evaluate(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        avg: str = None,
        pos_label: Union[int, str] = -1,
        console_out: bool = True,
        secondary_scoring: str = None,
        strength: int = 3,
    ) -> dict:
        """
        @param:
            x_test, y_test: Data to evaluate model

            avg: average to use for precision and recall score (e.g.: "micro", None, "weighted", "binary")
            pos_label: if avg="binary", pos_label says which class to score. pos_label is used by s_score/l_score

            console_out: shall the result be printed into the console

            secondary_scoring: weights the scoring (only for 's_score'/'l_score')
            strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for 's_score'/'l_score')
        """
        logging.debug("evaluation started...")
        pred = self.model.predict(x_test)

        # Calculate Accuracy, Precision and Recall Metrics
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average=avg, pos_label=pos_label)
        recall = recall_score(y_test, pred, average=avg, pos_label=pos_label)
        s_score = s_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        l_score = l_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)

        if console_out:
            print("accuracy: ", accuracy)
            print("precision: ", precision)
            print("recall: ", recall)

            print("classification report: ")
            print(classification_report(y_test, pred))

        self.test_score = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "s_score": s_score,
            "l_score": l_score,
        }

        logging.debug("... evaluation finished")
        return self.test_score

    def cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 3,
        avg: str = "macro",
        pos_label: Union[int, str] = -1,
        return_estimator: bool = False,
        console_out: bool = True,
        return_as_dict: bool = False,
        secondary_scoring: str = None,
        strength: int = 3,
    ) -> Union[dict[str, list], pd.DataFrame]:
        """
        @param:
            X, y: data to cross validate on
            cv_num: number of different splits

            avg: average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label: if avg="binary", pos_label says which class to score. pos_label is used by s_score/l_score

            return_estimator: if the estimator from the different splits shall be returned (suggestion: return_as_dict = True)

            console_out: shall the result be printed into the console
            return_as_dict: True: return scores as a dict, False: return scores as a pandas DataFrame

            secondary_scoring: weights the scoring (only for 's_score'/'l_score')
            strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for 's_score'/'l_score')

        @return:
            depending on "return_as_dict"
            the scores will be saved in self.cv_scores as dict (WARNING: return_estimator=True increases object size)
        """
        logging.debug("starting to cross validate...")

        precision_scorer = make_scorer(precision_score, average=avg, pos_label=pos_label)
        recall_scorer = make_scorer(recall_score, average=avg, pos_label=pos_label)
        s_scorer = make_scorer(s_scoring, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        l_scorer = make_scorer(l_scoring, strength=strength, scoring=secondary_scoring, pos_label=pos_label)

        if avg == "binary":
            scorer = {
                f"precision ({avg}, label={pos_label})": precision_scorer,
                f"recall ({avg}, label={pos_label})": recall_scorer,
                "accuracy": "accuracy",
                "s_score": s_scorer,
                "l_score": l_scorer,
            }
        else:
            scorer = {
                f"precision ({avg})": precision_scorer,
                f"recall ({avg})": recall_scorer,
                "accuracy": "accuracy",
                "s_score": s_scorer,
                "l_score": l_scorer,
            }

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

    def cross_validation_small_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sampling: str = "ros",
        vectorizer: str = "tfidf",
        scaler: str = "standard",
        avg: str = "macro",
        pos_label: Union[int, str] = -1,
        leave_loadbar: bool = True,
        console_out: bool = True,
        secondary_scoring: str = None,
        strength: int = 3,
    ) -> dict[str, list]:
        """
        Cross validation for small datasets (recommended for datasets with less than 150 datapoints)

        @param:
            X, y: data to cross validate on
            sampling: type of "data.sampling.Sampler" class or None for no sampling
            vectorizer: type of "data.embeddings.Embeddings_builder" for automatic string column vectorizing
            scaler: type of "data.scaler.Scaler" for scaling the data

            avg: average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label: if avg="binary", pos_label says which class to score. pos_label is used by s_score/l_score

            leave_loadbar: shall the loading bar of the training be visible after training (True - load bar will still be visible)
            console_out: shall the result be printed into the console

            secondary_scoring: weights the scoring (only for 's_score'/'l_score')
            strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for 's_score'/'l_score')

        @return:
            dictionary with "accuracy", "precision", "recall", "s_score", "l_score", "avg train score", "avg train time"
        """
        logging.debug("starting to cross validate...")

        predictions = []
        true_values = []
        t_scores = []
        t_times = []

        # auto-detect data types
        X = X.convert_dtypes()
        string_columns = X.select_dtypes(include="string").columns

        sampling_problems = ["QDA", "LDA", "LR", "MLPC", "LSVC"]

        if sampling == "SMOTE" and self.model_type in sampling_problems:
            if console_out:
                print(self.model_type+" does not work with sampling='SMOTE' --> going on with sampling='ros'")
            sampling = "ros"

        elif sampling in ["nm", "tl"] and self.model_type in sampling_problems:
            if console_out:
                print(self.model_type+f" does not work with sampling='{sampling}' --> going on with sampling='rus'")
            sampling = "rus"

        eb = Embeddings_builder(vec=vectorizer, console_out=False)
        
        for idx in tqdm(X.index, desc=self.model_name, leave=leave_loadbar):
            x_train = X.drop(idx)
            y_train = y.drop(idx)
            x_test = X.loc[[idx]]
            y_test = y.loc[idx]

            for col in string_columns:
                x_train = pd.concat([x_train, eb.vectorize(x_train[col], train_on=True)], axis=1)
                x_test = pd.concat([x_test, eb.vectorize(x_test[col], train_on=False)], axis=1)

            x_train = x_train.drop(columns=string_columns)
            x_test = x_test.drop(columns=string_columns)

            if scaler != None:
                sc = Scaler(scaler=scaler, console_out=False)
                x_train = sc.scale(x_train, train_on=True)
                x_test = sc.scale(x_test, train_on=False)

            if sampling != None:
                sampler = Sampler(algorithm=sampling)
                x_train, y_train = sampler.sample(x_train, y_train)

            train_score, train_time = self.train(x_train, y_train, console_out=False)
            prediction = self.model.predict(x_test)

            predictions.append(prediction)
            true_values.append(y_test)
            t_scores.append(train_score)
            t_times.append(train_time)

        accuracy = accuracy_score(true_values, predictions)
        precision = precision_score(true_values, predictions, average=avg, pos_label=pos_label)
        recall = recall_score(true_values, predictions, average=avg, pos_label=pos_label)
        s_score = s_scoring(true_values, predictions, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        l_score = l_scoring(true_values, predictions, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        avg_train_score = mean(t_scores)
        avg_train_time = str(timedelta(seconds=round(sum(map(lambda f: int(f[0])*3600 + int(f[1])*60 + int(f[2]), map(lambda f: f.split(':'), t_times)))/len(t_times))))

        self.cv_scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "s_score": s_score,
            "l_score": l_score,
            "avg train score": avg_train_score,
            "avg train time": avg_train_time,
        }

        if console_out:
            print("classification report:")
            print(classification_report(true_values, predictions))

        return self.cv_scores

    def feature_importance(self) -> plt.show:
        """
        feature_importance() generates a matplotlib plot of the feature importance from self.model
        """
        if not self.trained:
            return "INFO: You have to first train the classifier before getting the feature importance"

        if self.model_type == "MLPC":
            importances = [np.mean(i) for i in self.model.coefs_[0]]  # MLP Classifier
        elif self.model_type in ["DTC", "RFC", "GBM", "CBC", "ABC", "ETC"]:
            importances = self.model.feature_importances_
        elif self.model_type in ["KNC", "GNB", "BNB", "GPC", "QDA", "BC"]:
            return self.model_name+" does not have a feature importance"
        else:
            importances = self.model.coef_[0]  # "normal"

        feature_importances = pd.Series(importances, index=self.feature_names)

        fig, ax = plt.subplots()
        if self.model_type in ["RFC", "GBM", "ETC"]:
            if self.model_type in ["RFC", "ETC"]:
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
        grid: dict = None,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        verbose: int = 0,
        rand_search: bool = True,
        n_iter_num: int = 75,
        console_out: bool = True,
        train_afterwards: bool = True,
        secondary_scoring: str = None,
        strength: int = 3,
    ):
        """
        @param:
            x_train: DataFrame with train features
            y_train: Series with labels

            grid: dictonary of parameters to tune (default: default parameter dictionary self.grid)

            scoring: metrics to evaluate the models
            avg: average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label: if avg="binary", pos_label says which class to score. Else pos_label is ignored (except: scoring='s_score'/'l_score')

            rand_search: True: RandomizedSearchCV, False: GridSearchCV
            n_iter_num: Combinations to try out if rand_search=True

            n_split_num: number of different splits
            n_repeats_num: number of repetition of one split

            verbose: log level (higher number --> more logs)
            console_out: output the the results of the different iterations
            train_afterwards: train the best model after finding it

            secondary_scoring: weights the scoring (only for scoring='s_score'/'l_score')
            strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for scoring='s_score'/'l_score')

        @return:
            set self.model = best model from search
        """
        if grid is None:
            grid = self.grid

        if console_out:
            print("grid: ", grid)
            print()

        if scoring == "precision":
            scoring = make_scorer(precision_score, average=avg, pos_label=pos_label)
        elif scoring == "recall":
            scoring = make_scorer(recall_score, average=avg, pos_label=pos_label)
        elif scoring == "s_score":
            scoring = make_scorer(s_scoring, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        elif scoring == "l_score":
            scoring = make_scorer(l_scoring, strength=strength, scoring=secondary_scoring, pos_label=pos_label)

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
        print()
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print()

        if train_afterwards:
            logging.debug("starting to train best model...")
            self.train(x_train, y_train, console_out=console_out)
            logging.debug("... best model trained")
