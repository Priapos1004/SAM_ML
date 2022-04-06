import logging
from typing import Union

import pandas as pd
from tqdm.notebook import tqdm

from .DecisionTreeClassifier import DTC
from .LogisticRegression import LR
from .main_classifier import Classifier
from .MLPClassifier import MLPC
from .RandomForestClassifier import RFC
from .SupportVectorClassifier import SVC


class CTest:
    def __init__(self, models: list[Classifier] = [DTC(), LR(), MLPC(), RFC(), SVC()]):
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
            self.models[key].train(x_train, y_train, console_out=False)
            score = self.models[key].evaluate(
                x_test, y_test, avg=avg, pos_label=pos_label, console_out=False
            )
            self.scores[key] = score

        logging.debug("... models evaluated")

        return self.scores

    def output_scores_as_pd(self, console_out: bool = True) -> pd.DataFrame:
        scores = pd.DataFrame(self.scores).transpose()

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
    ) -> Classifier:
        """
        @param:
            scoring - "accuracy" / "precision" / "recall"
            avg - average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored

            rand_search - True: RandomizedSearchCV, False: GridSearchCV
            n_iter_num - Combinations to try out if rand_search=True

        @return:
            prints parameters and metrics of best model
            saves best model in self.best_model
            returns best model
        """
        if self.scores == {}:
            self.eval_models(
                x_train, y_train, x_test, y_test, avg=avg, pos_label=pos_label
            )
            self.output_scores_as_pd()
            print()
        else:
            print(
                "-> using already created scores for the models. Please run 'eval_models()' again if something changed with the data"
            )

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
        self.models[best_model_type].hyperparameter_tuning(
            x_train,
            y_train,
            scoring=scoring,
            train_afterwards=True,
            avg=avg,
            pos_label=pos_label,
            rand_search=rand_search,
            n_iter_num=n_iter_num,
        )
        print("... hyperparameter tuning finished")
        print()

        logging.debug("Set self.best_model = hyperparameter tuned model")
        self.best_model = self.models[best_model_type]

        self.best_model.evaluate(x_test, y_test, avg=avg, pos_label=pos_label)

        return self.best_model
