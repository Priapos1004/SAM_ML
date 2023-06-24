import os
import sys
import warnings
from copy import copy
from typing import Union

import pandas as pd

# to deactivate pygame promt 
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
from pkg_resources import resource_filename
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from sam_ml.config import setup_logger
from sam_ml.data import Embeddings_builder, Sampler, Scaler, Selector

from .AdaBoostClassifier import ABC
from .BaggingClassifier import BC
from .BernoulliNB import BNB
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
from .main_pipeline import Pipeline
from .MLPClassifier import MLPC
from .QuadraticDiscriminantAnalysis import QDA
from .RandomForestClassifier import RFC
from .SupportVectorClassifier import SVC
from .XGBoostClassifier import XGBC

logger = setup_logger(__name__)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affects subprocesses


class CTest:
    """ AutoML class """

    def __init__(self, models: Union[str, list[Classifier]] = "search", vectorizer: Union[str, Embeddings_builder] = None, scaler: Union[str, Scaler] = None, selector: Union[str, Selector] = None, sampler: Union[str, Sampler] = None):
        """
        @params:
            models:

                - list of Wrapperclass models from this library

                - 'all': use all Wrapperclass models (18+ models)

                - 'basic': use basic Wrapperclass models (8 models) (LogisticRegression, MLP Classifier, LinearSVC, DecisionTreeClassifier, RandomForestClassifier, SVC, Gradientboostingmachine, KNeighborsClassifier)

            vectorizer: type of "data.embeddings.Embeddings_builder" or Embeddings_builder class object for automatic string column vectorizing (None for no vectorizing)
            scaler: type of "data.scaler.Scaler" or Scaler class object for scaling the data (None for no scaling)
            selector: type of "data.feature_selection.Selector" or Selector class object for feature selection (None for no selecting)
            sampling: type of "data.sampling.Sampler" or Sampler class object for sampling the train data (None for no sampling)
        """
        self.__models_input = models

        if type(models) == str:
            models = self.model_combs(models)

        self.models: dict = {}
        for i in range(len(models)):
            self.models[models[i].model_name] = Pipeline(vectorizer,  scaler, selector, sampler, models[i], models[i].model_name+" (pipeline)")

        self._vectorizer = vectorizer
        self._scaler = scaler
        self._selector = selector
        self._sampler = sampler
        self.best_model: Pipeline
        self.scores: dict = {}

    def __repr__(self) -> str:
        params: str = ""

        if type(self.__models_input) == str:
            params += f"models='{self.__models_input}', "
        else:
            params += "models=["
            for model in self.__models_input:
                params += f"\n    {model.__str__()},"
            params += "],\n"

        if type(self._vectorizer) == str:
            params += f"vectorizer='{self._vectorizer}'"
        elif type(self._vectorizer) == Embeddings_builder:
            params += f"vectorizer={self._vectorizer.__str__()}"
        else:
            params += f"vectorizer={self._vectorizer}"
        
        params += ", "

        if type(self._scaler) == str:
            params += f"scaler='{self._scaler}'"
        elif type(self._scaler) == Scaler:
            params += f"scaler={self._scaler.__str__()}"
        else:
            params += f"scaler={self._scaler}"

        params += ", "

        if type(self._selector) == str:
            params += f"selector='{self._selector}'"
        elif type(self._selector) == Selector:
            params += f"selector={self._selector.__str__()}"
        else:
            params += f"selector={self._selector}"

        params += ", "

        if type(self._sampler) == str:
            params += f"sampler='{self._sampler}'"
        elif type(self._sampler) == Sampler:
            params += f"sampler={self._sampler.__str__()}"
        else:
            params += f"sampler={self._sampler}"

        return f"CTest({params})"

    def remove_model(self, model_name: str):
        del self.models[model_name]

    def add_model(self, model: Classifier):
        self.models[model.model_name] = Pipeline(self._vectorizer, self._scaler, self._selector, self._sampler, model, model.model_name+" (pipeline)")

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
                RFC(),
                SVC(),
                GBM(),
                
                ABC(model_name="AdaBoostClassifier (DTC based)"),
                ABC(
                    estimator=RandomForestClassifier(max_depth=5, random_state=42),
                    model_name="AdaBoostClassifier (RFC based)",
                ),
                ABC(
                    estimator=LogisticRegression(),
                    model_name="AdaBoostClassifier (mixed based)",
                ),
                KNC(),
                ETC(),
                GNB(),
                BNB(),
                GPC(),
                BC(model_name="BaggingClassifier (DTC based)"),
                BC(
                    estimator=RandomForestClassifier(max_depth=5, random_state=42),
                    model_name="BaggingClassifier (RFC based)",
                ),
                BC(
                    estimator=LogisticRegression(),
                    model_name="BaggingClassifier (mixed based)",
                ),
                XGBC(),
            ]
        elif kind == "basic":
            models = [
                LR(),
                MLPC(),
                LSVC(),
                DTC(),
                RFC(),
                SVC(),
                GBM(),
                KNC(),
            ]
        else:
            print(f"Cannot find model combination '{kind}' --> using all models")
            models = self.model_combs("all")

        return models

    def __finish_sound(self):
        """ little function to play a microwave sound """
        filepath = resource_filename(__name__, 'microwave_finish_sound.mp3')
        pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()

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
            logger.info("KeyboardInterrupt - output interim result")
            return self.scores

    def eval_models_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 5,
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
            logger.info("KeyboardInterrupt - output interim result")
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
            logger.warning("no scores are created -> use 'eval_models()'/'eval_models_cv()' to create scores")
            scores = None

        return scores

    def find_best_model_randomCV(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_trails: int = 5,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = -1,
        secondary_scoring: str = None,
        strength: int = 3,
        small_data_eval: bool = False,
        cv_num: int = 3,
    ) -> dict:
        for key in tqdm(self.models.keys(), desc="randomCVsearch"):
            best_hyperparameters, best_score = self.models[key].randomCVsearch(x_train, y_train, n_trails, scoring, avg, pos_label, secondary_scoring, strength, small_data_eval, cv_num, True)
            logger.info(f"{self.models[key].model_name} - score: {best_score} ({scoring}) - parameters: {best_hyperparameters}")
            model_best = copy(self.models[key])
            model_best.set_params(**best_hyperparameters)
            train_score, train_time = model_best.train(x_train, y_train, console_out=False)
            scores = model_best.evaluate(x_test, y_test, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength, console_out=False)
            
            scores["train_time"] = train_time
            scores["train_score"] = train_score
            scores["best_score (rCVs)"] = best_score
            scores["best_hyperparameters (rCVs)"] = best_hyperparameters
            self.scores[key] = scores
        sorted_scores = self.output_scores_as_pd(sort_by=[scoring, "s_score", "train_time"], console_out=False)
        best_model_type = sorted_scores.iloc[0].name
        best_model_value = sorted_scores.iloc[0][scoring]
        best_model_hyperparameters = sorted_scores.iloc[0]["best_hyperparameters (rCVs)"]
        logger.info(f"best model type {best_model_type} - {scoring}: {best_model_value} - parameters: {best_model_hyperparameters}")
        self.__finish_sound()
        return self.scores


    # def find_best_model(
    #     self,
    #     x_train: pd.DataFrame,
    #     y_train: pd.Series,
    #     x_test: pd.DataFrame,
    #     y_test: pd.Series,
    #     cv_kind: str = "no",
    #     scoring: str = "accuracy",
    #     avg: str = "macro",
    #     pos_label: Union[int, str] = -1,
    #     rand_search: bool = True,
    #     n_iter_num: int = 75,
    #     cv_num: int = 10,
    #     console_out: bool = False,
    #     secondary_scoring: str = None,
    #     strength: int = 3,
    # ) -> Pipeline:
    #     """
    #     @param:
    #         cv_kind: which kind of cross validation shall be used to create the scores to find the best model type
    #             'no': use eval_models on x_train, y_train, x_test, y_test
    #             'small': use eval_models_cv with small_data_eval=True on x_train, y_train
    #             'multi': use eval_models_cv with small_data_eval=False on x_train, y_train
    #         scoring: "accuracy" / "precision" / "recall" / "s_score" / "l_score"

    #         avg: average to use for precision and recall score (e.g. "micro", "weighted", "binary")
    #         pos_label: if avg="binary", pos_label says which class to score. Else pos_label is ignored (except scoring='s_score'/'l_score')
    #         rand_search: True: RandomizedSearchCV, False: GridSearchCV
    #         n_iter_num: Combinations to try out if rand_search=True
    #         cv_num: number of different splits
    #         console_out: outputs intermidiate results into the console
    #         secondary_scoring: weights the scoring (only for scoring='s_score'/'l_score')
    #         strength: higher strength means a higher weight for the prefered secondary_scoring/pos_label (only for scoring='s_score'/'l_score')

    #     @return:
    #         - prints parameters and metrics of best model
    #         - saves best model in self.best_model
    #         - returns best model
    #     """

    #     if cv_kind == "no":
    #         logger.debug("creating scores using 'eval_models()'")
    #         self.eval_models(x_train, y_train, x_test, y_test, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
    #     elif cv_kind == "small":
    #         logger.debug("creating scores using 'eval_models_cv(small_data_eval=True)'")
    #         self.eval_models_cv(x_train, y_train, avg=avg, pos_label=pos_label, small_data_eval=True, secondary_scoring=secondary_scoring, strength=strength)
    #     elif cv_kind == "multi":
    #         logger.debug("creating scores using 'eval_models_cv(small_data_eval=False)'")
    #         self.eval_models_cv(x_train, y_train, avg=avg, pos_label=pos_label, small_data_eval=False, secondary_scoring=secondary_scoring, strength=strength, cv_num=cv_num)
    #     else:
    #         logger.error(f"wrong input '{cv_kind}' for cv_kind")
    #         return

    #     sorted_scores = self.output_scores_as_pd(sort_by=[scoring, "s_score"])
    #     best_model_type = sorted_scores.iloc[0].name
    #     best_model_value = sorted_scores.iloc[0][scoring]

    #     print()
    #     print(f"best model type ({scoring}): ", best_model_type, " - ", best_model_value)
    #     print()

    #     logger.info(f"hyperparametertuning for best model type (rand_search = {rand_search}) - started")
    #     self.models[best_model_type].gridsearch(
    #         x_train,
    #         y_train,
    #         scoring=scoring,
    #         train_afterwards=True,
    #         avg=avg,
    #         pos_label=pos_label,
    #         rand_search=rand_search,
    #         n_iter_num=n_iter_num,
    #         cv_num=cv_num,
    #         console_out=console_out,
    #         secondary_scoring=secondary_scoring,
    #         strength=strength,
    #     )
    #     logger.info(f"hyperparametertuning for best model type (rand_search = {rand_search}) - finished")

    #     # Set self.best_model = hyperparameter tuned model
    #     self.best_model = self.models[best_model_type]

    #     self.best_model.evaluate(x_test, y_test, avg=avg, pos_label=pos_label)
    #     self.best_model.feature_importance()
    #     self.__finish_sound()
    #     return self.best_model
