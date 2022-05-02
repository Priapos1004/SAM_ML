import logging
from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm.notebook import tqdm

from .AdaBoostClassifier import ABC
from .BernoulliNB import BNB
from .CatBoostClassifier import CBC
from .DecisionTreeClassifier import DTC
from .ExtraTreesClassifier import ETC
from .GaussianNB import GNB
from .GaussianProcessClassifier import GPC
from .GradientBoostingMachine import GBM
from .KNeighborsClassifier import KNC
from .LinearDiscriminantAnalysis import LDA
from .LogisticRegression import LR
from .main_classifier import Classifier
from .MLPClassifier import MLPC
from .QuadraticDiscriminantAnalysis import QDA
from .RandomForestClassifier import RFC
from .SupportVectorClassifier import SVC


class CTest:
    def __init__(
        self,
        models: list[Classifier] = [
            DTC(),
            LR(),
            MLPC(),
            RFC(),
            SVC(model_name="SupportVectorMachine (linear-kernel)"),
            SVC(kernel="rbf", model_name="SupportVectorMachine (rbf-kernel)"),
            GBM(),
            CBC(),
            ABC(model_name="AdaBoostClassifier (DTC based)"),
            ABC(
                base_estimator=RandomForestClassifier(max_depth=5),
                model_name="AdaBoostClassifier (RFC based)",
            ),
            KNC(),
            ETC(),
            GNB(),
            BNB(),
            GPC(),
            QDA(),
            LDA(),
        ],
    ):
        self.models: dict = {}
        for i in range(len(models)):
            self.models[models[i].model_name] = models[i]

        self.best_model: Classifier = None
        self.scores: dict = {}

    def eval_models(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
    ) -> dict[dict]:
        """
        @param:
            x_train, y_train, x_test, y_test - Data to train and evaluate models
            avg - average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored

        @return:
            saves metrics in dict self.scores and also outputs them
        """
        logging.debug("starting to evaluate models...")
        for key in tqdm(self.models.keys()):
            tscore, ttime = self.models[key].train(x_train, y_train, console_out=False)
            score = self.models[key].evaluate(
                x_test, y_test, avg=avg, pos_label=pos_label, console_out=False
            )
            score["train_score"] = tscore
            score["train_time"] = ttime
            self.scores[key] = score

        logging.debug("... models evaluated")

        return self.scores

    def eval_models_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 3,
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
    ) -> dict[dict]:
        """
        @param:
            X, y - Data to train and evaluate models on
            cv_num - number of different splits
            avg - average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored

        @return:
            saves metrics in dict self.scores and also outputs them
        """
        logging.debug("starting to evaluate models...")
        for key in tqdm(self.models.keys()):
            self.models[key].cross_validation(
                X, y, cv_num=cv_num, avg=avg, pos_label=pos_label, console_out=False
            )
            score = self.models[key].cv_scores["average"]
            self.scores[key] = {
                "accuracy": score["test_accuracy"],
                "precision": score[list(score.keys())[2]],
                "recall": score[list(score.keys())[4]],
            }

        logging.debug("... models evaluated")

        return self.scores

    def output_scores_as_pd(
        self, sort_by: Union[str, list[str]] = "index", console_out: bool = True
    ) -> pd.DataFrame:
        """
        @param:
            sorted_by:
                'index' - sort index ascending=True
                'precision'/'recall'/'accuracy'/'train_score'/'train_time' - sort by these columns ascending=False
                e.g. ['precision', 'recall'] - sort first by 'precision' and then by 'recall'
        """
        if sort_by == "index":
            scores = pd.DataFrame(self.scores).transpose().sort_index(ascending=True)
        else:
            scores = (
                pd.DataFrame(self.scores)
                .transpose()
                .sort_values(by=sort_by, ascending=False)
            )

        if console_out:
            print(scores)

        return scores

    def find_best_model(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        rand_search: bool = True,
        n_iter_num: int = 75,
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        console_out: bool = False,
    ) -> Classifier:
        """
        @param:
            scoring - "accuracy" / "precision" / "recall"
            avg - average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored

            rand_search - True: RandomizedSearchCV, False: GridSearchCV
            n_iter_num - Combinations to try out if rand_search=True

            n_split_num - number of different splits
            n_repeats_num - number of repetition of one split

            console_out - outputs intermidiate results into the console

        @return:
            prints parameters and metrics of best model
            saves best model in self.best_model
            returns best model
        """
        if self.scores == {}:
            print(
                "no scores are already created -> creating scores using 'eval_models()'"
            )
            self.eval_models(
                x_train, y_train, x_test, y_test, avg=avg, pos_label=pos_label
            )
            self.output_scores_as_pd(sort_by=scoring)
            print()
        else:
            print(
                "-> using already created scores for the models. Please run 'eval_models()'/'eval_models_cv()' again if something changed with the data"
            )
            print()

        sorted_scores = sorted(
            self.scores.items(), key=lambda x: x[1][scoring], reverse=True
        )
        best_model_type = sorted_scores[0][0]
        best_model_value = sorted_scores[0][1][scoring]

        print(
            f"best model type ({scoring}): ", best_model_type, " - ", best_model_value
        )

        print(
            "starting to hyperparametertune best model type (rand_search = ",
            rand_search,
            ")...",
        )
        print()
        self.models[best_model_type].hyperparameter_tuning(
            x_train,
            y_train,
            scoring=scoring,
            train_afterwards=True,
            avg=avg,
            pos_label=pos_label,
            rand_search=rand_search,
            n_iter_num=n_iter_num,
            n_repeats_num=n_repeats_num,
            n_split_num=n_split_num,
            console_out=console_out,
        )
        print()
        print("... hyperparameter tuning finished")
        print()

        logging.debug("Set self.best_model = hyperparameter tuned model")
        self.best_model = self.models[best_model_type]

        self.best_model.evaluate(x_test, y_test, avg=avg, pos_label=pos_label)

        return self.best_model
