import os
import sys
import warnings
from typing import Union

import pandas as pd
from pkg_resources import resource_filename
from playsound import playsound
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

from .AdaBoostClassifier import ABC
from .BaggingClassifier import BC
from .BernoulliNB import BNB
from .CatBoostClassifier import CBC
from .DecisionTreeClassifier import DTC
from .ExtraTreesClassifier import ETC
from .GaussianNB import GNB
from .GaussianProcessClassifier import GPC
from .GradientBoostingMachine import GBM
from .KNeighborsClassifier import KNC
from .LinearDiscriminantAnalysis import LDA
from .LinearSupportVectorClassifier import LSVC
from .LogisticRegression import LR
from .main_classifier import Classifier
from .MLPClassifier import MLPC
from .QuadraticDiscriminantAnalysis import QDA
from .RandomForestClassifier import RFC
from .SupportVectorClassifier import SVC

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affects subprocesses


class CTest:
    """ AutoML class """

    def __init__(self, models: Union[str, list[Classifier]] = "all"):
        """
        @params:
            models:
                list of Wrapperclass models from sam_ml library
                'all': use all Wrapperclass models (18 models) from sam_ml library
                'basic': use basic Wrapperclass models (9 models) from sam_ml library (LogisticRegression, MLP Classifier, LinearSVC, DecisionTreeClassifier, RandomForestClassifier, SVC, Gradientboostingmachine, AdaboostClassifier, KNeighborsClassifier)
        """
        if type(models) == str:
            models = self.model_combs(models)

        self.models: dict = {}
        for i in range(len(models)):
            self.models[models[i].model_name] = models[i]

        self.best_model: Classifier
        self.scores: dict = {}

    def remove_model(self, model_name: str):
        del self.models[model_name]

    def add_model(self, model: Classifier):
        self.models[model.model_name] = model

    def model_combs(self, kind: str):
        """
        @params:
            kind:
                "all": use all models
                "basic": use a simple combination (LogisticRegression, MLP Classifier, LinearSVC, DecisionTreeClassifier, RandomForestClassifier, SVC, Gradientboostingmachine, AdaboostClassifier, KNeighborsClassifier)
        """
        if kind == "all":
            models = [
                LR(),
                QDA(),
                LDA(),
                MLPC(),
                LSVC(),
                DTC(),
                #CBC(),
                RFC(),
                SVC(model_name="SupportVectorClassifier (rbf-kernel)"),
                GBM(),
                
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
                BC(model_name="BaggingClassifier (DTC based)"),
                BC(
                    base_estimator=RandomForestClassifier(max_depth=5),
                    model_name="BaggingClassifier (RFC based)",
                ),
            ]
        elif kind == "basic":
            models = [
                LR(),
                MLPC(),
                LSVC(),
                DTC(),
                RFC(),
                SVC(model_name="SupportVectorClassifier (rbf-kernel)"),
                GBM(),
                ABC(model_name="AdaBoostClassifier (DTC based)"),
                KNC(),
            ]
        else:
            print(f"Cannot find model combination '{kind}' --> using all models")
            models = self.model_combs("all")

        return models

    def __finish_sound(self):
        """ little function to play a microwave sound """
        filepath = resource_filename(__name__, 'microwave_finish_sound.mp3')
        playsound(filepath)

    def eval_models(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        avg: str = "macro",
        pos_label: Union[int, str] = -1,
        secondary_scoring: str = None,
        strength: int = 3,
    ) -> dict[str, dict]:
        """
        @param:
            x_train, y_train, x_test, y_test: Data to train and evaluate models

            avg: average to use for precision and recall score (e.g. "micro", "weighted", "binary")    
            pos_label: if avg="binary", pos_label says which class to score. pos_label is used by s_score/l_score

            secondary_scoring: weights the scoring (only for 's_score'/'l_score')
            strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for 's_score'/'l_score')

        @return:
            saves metrics in dict self.scores and also outputs them
        """
        try:
            for key in tqdm(self.models.keys(), desc="Crossvalidation"):
                tscore, ttime = self.models[key].train(x_train, y_train, console_out=False)
                score = self.models[key].evaluate(
                    x_test, y_test, avg=avg, pos_label=pos_label, console_out=False, secondary_scoring=secondary_scoring, strength=strength,
                )
                score["train_score"] = tscore
                score["train_time"] = ttime
                self.scores[key] = score

            self.__finish_sound()
            return self.scores

        except KeyboardInterrupt:
            print("KeyboardInterrupt - output interim result")
            return self.scores

    def eval_models_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 3,
        avg: str = "macro",
        pos_label: Union[int, str] = -1,
        small_data_eval: bool = False,
        secondary_scoring: str = None,
        strength: int = 3,
    ) -> dict[str, dict]:
        """
        @param:
            X, y: Data to train and evaluate models on
            cv_num: number of different splits (ignored if small_data_eval=True)

            avg: average to use for precision and recall score (e.g. "micro", "weighted", "binary")
            pos_label: if avg="binary", pos_label says which class to score. pos_label is used by s_score/l_score
            
            small_data_eval: if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)

            secondary_scoring: weights the scoring (only for 's_score'/'l_score')
            strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for 's_score'/'l_score')

        @return:
            saves metrics in dict self.scores and also outputs them
        """

        try:
            for key in tqdm(self.models.keys(), desc="Crossvalidation"):
                if small_data_eval:
                    self.models[key].cross_validation_small_data(
                        X, y, avg=avg, pos_label=pos_label, console_out=False, leave_loadbar=False, secondary_scoring=secondary_scoring, strength=strength,
                    )
                else:
                    self.models[key].cross_validation(
                        X, y, cv_num=cv_num, avg=avg, pos_label=pos_label, console_out=False, secondary_scoring=secondary_scoring, strength=strength,
                    )
                self.scores[key] = self.models[key].cv_scores
            self.__finish_sound()
            return self.scores

        except KeyboardInterrupt:

            print("KeyboardInterrupt - output interim result")
            return self.scores

    def output_scores_as_pd(self, sort_by: Union[str, list[str]] = "index", console_out: bool = True) -> pd.DataFrame:
        """
        @param:
            sorted_by:
                'index': sort index ascending=True
                'precision'/'recall'/'accuracy'/'train_score'/'train_time': sort by these columns ascending=False

                e.g. ['precision', 'recall'] - sort first by 'precision' and then by 'recall'
        """
        if self.scores != {}:
            if sort_by == "index":
                scores = pd.DataFrame.from_dict(self.scores, orient="index").sort_index(ascending=True)
            else:
                scores = (
                    pd.DataFrame.from_dict(self.scores, orient="index")
                    .sort_values(by=sort_by, ascending=False)
                )

            if console_out:
                print(scores)
        else:
            print("WARNING: no scores are created -> use 'eval_models()'/'eval_models_cv()' to create scores")
            scores = None

        return scores

    def find_best_model(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = -1,
        rand_search: bool = True,
        n_iter_num: int = 75,
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        console_out: bool = False,
        secondary_scoring: str = None,
        strength: int = 3,
    ) -> Classifier:
        """
        @param:
            scoring: "accuracy" / "precision" / "recall" / "s_score" / "l_score"

            avg: average to use for precision and recall score (e.g. "micro", "weighted", "binary")
            pos_label: if avg="binary", pos_label says which class to score. Else pos_label is ignored (except scoring='s_score'/'l_score')
            rand_search: True: RandomizedSearchCV, False: GridSearchCV
            n_iter_num: Combinations to try out if rand_search=True
            n_split_num: number of different splits
            n_repeats_num: number of repetition of one split
            console_out: outputs intermidiate results into the console
            secondary_scoring: weights the scoring (only for scoring='s_score'/'l_score')
            strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for scoring='s_score'/'l_score')

        @return:
            - prints parameters and metrics of best model
            - saves best model in self.best_model
            - returns best model
        """
        new_scores = (self.scores == {})
        if new_scores:
            print("no scores are already created -> creating scores using 'eval_models()'")
            self.eval_models(x_train, y_train, x_test, y_test, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
        else:
            print("-> using already created scores for the models. Please run 'eval_models()'/'eval_models_cv()' again if something changed with the data")

        sorted_scores = self.output_scores_as_pd(sort_by=[scoring, "s_score"], console_out=new_scores)
        print()
        best_model_type = sorted_scores.iloc[0].name
        best_model_value = sorted_scores.iloc[0][scoring]

        print(f"best model type ({scoring}): ", best_model_type, " - ", best_model_value)

        print(f"starting to hyperparametertune best model type (rand_search = {rand_search})...",)
        print()

        self.models[best_model_type].gridsearch(
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
            secondary_scoring=secondary_scoring,
            strength=strength,
        )

        print()
        print("... hyperparameter tuning finished")
        print()

        # Set self.best_model = hyperparameter tuned model
        self.best_model = self.models[best_model_type]

        self.best_model.evaluate(x_test, y_test, avg=avg, pos_label=pos_label)
        self.__finish_sound()
        return self.best_model
