import os
import sys
import time
import warnings
from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd

# to deactivate pygame promt 
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
from pkg_resources import resource_filename
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from sam_ml.config import get_sound_on, setup_logger
from sam_ml.data import Embeddings_builder, Sampler, SamplerPipeline, Scaler, Selector

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

    def __init__(self, models: str | list[Classifier] = "all", vectorizer: str | Embeddings_builder | None | list[str | Embeddings_builder | None] = None, scaler: str | Scaler | None  | list[str | Scaler | None] = None, selector: str | Selector | None  | list[str | Selector | None] = None, sampler: str | Sampler | SamplerPipeline | None  | list[str | Sampler | SamplerPipeline | None] = None):
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

        if type(vectorizer) in (str, Embeddings_builder) or vectorizer is None:
            vectorizer = [vectorizer]

        if type(scaler) in (str, Scaler) or scaler is None:
            scaler = [scaler]

        if type(selector) in (str, Selector) or selector is None:
            selector = [selector]

        if type(sampler) in (str, Sampler) or sampler is None:
            sampler = [sampler]

        self._vectorizer = vectorizer
        self._scaler = scaler
        self._selector = selector
        self._sampler = sampler

        self.models: dict = {}
        for model in models:
            self.add_model(model)

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
        for vec in self._vectorizer:
            for scal in self._scaler:
                for sel in self._selector:
                    for sam in self._sampler:
                        model_pipe_name = model.model_name+f" (vec={vec}, scaler={scal}, selector={sel}, sampler={sam})"
                        self.models[model_pipe_name] = Pipeline(vec,  scal, sel, sam, model, model_pipe_name)

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
                    model_name="AdaBoostClassifier (LR based)",
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
                    model_name="BaggingClassifier (LR based)",
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
        if get_sound_on():
            filepath = resource_filename(__name__, 'microwave_finish_sound.mp3')
            pygame.mixer.init()
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()

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
            strength: higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for 's_score'/'l_score')

        @return:
            saves metrics in dict self.scores and also outputs them
        """
        try:
            for key in tqdm(self.models.keys(), desc="Evaluation"):
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
            strength: higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for 's_score'/'l_score')

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
        leave_loadbar: bool = True,
    ) -> dict:
        """
        @params:
            x_train: DataFrame with train features
            y_train: Series with train labels
            x_test: DataFrame with test features
            y_test: Series with test labels

            n_trails: number of parameter sets to test

            scoring: metrics to evaluate the models
            avg: average to use for precision and recall score (e.g. "micro", "weighted", "binary")
            pos_label: if avg="binary", pos_label says which class to score. Else pos_label is ignored (except scoring='s_score'/'l_score')
            secondary_scoring: weights the scoring (only for scoring='s_score'/'l_score')
            strength: higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for scoring='s_score'/'l_score')

            small_data_eval: if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)

            cv_num: number of different splits per crossvalidation (only used when small_data_eval=False)

            leave_loadbar: shall the loading bar of the randomCVsearch of each individual model be visible after training (True - load bar will still be visible)
        """
        for key in tqdm(self.models.keys(), desc="randomCVsearch"):
            best_hyperparameters, best_score = self.models[key].randomCVsearch(x_train, y_train, n_trails=n_trails, scoring=scoring, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength, small_data_eval=small_data_eval, cv_num=cv_num, leave_loadbar=leave_loadbar)
            logger.info(f"{self.models[key].model_name} - score: {best_score} ({scoring}) - parameters: {best_hyperparameters}")
            if best_hyperparameters != {}:
                model_best = self.models[key].get_deepcopy()
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
    
    def find_best_model_mass_search(self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_trails: int = 10,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = -1,
        secondary_scoring: str = None,
        strength: int = 3,
        leave_loadbar: bool = True,
        save_results_path: str | None = "find_best_model_mass_search_results.csv",
    ) -> dict:
        model_dict = {}
        for key in self.models.keys():
            model = self.models[key]
            configs = model.get_random_configs(n_trails)
            try:
                for config in configs:
                    model_new = model.get_deepcopy()
                    model_new = model_new.set_params(**config)
                    if model_new.model_type != "XGBC":
                        model_new = model_new.set_params(**{"warm_start": True})
                    model_name = f"{key} {dict(config)}"
                    model_dict[model_name] = model_new
            except:
                logger.warning(f"modeltype in '{key}' is not supported for this search -> will be skipped")

        total_model_num = len(model_dict)
        logger.info(f"total number of models: {total_model_num}")
        split_num = int(np.log2(total_model_num))+1
        split_size =int(1/split_num*len(x_train))
        logger.info(f"split number: {split_num-1}, split_size (x_train): {split_size}")
        if split_size < 300:
            raise RuntimeError(f"not enough data for the amout of models. Data per split should be over 300, but {split_size} < 300")

        # shuffle x_train/y_train
        x_train = x_train.sample(frac=1, random_state=42)
        y_train = y_train.sample(frac=1, random_state=42)

        for split_idx in tqdm(range(split_num-1), desc="splits"):
            x_train_train = x_train[split_idx*split_size:(split_idx+1)*split_size]
            x_train_test = x_train[(split_idx+1)*split_size:]
            y_train_train = y_train[split_idx*split_size:(split_idx+1)*split_size]
            y_train_test = y_train[(split_idx+1)*split_size:]
            logger.info(f"split {split_idx+1}: length x_train/y_train {len(x_train_train)}/{len(y_train_train)}, length x_test/y_test {len(x_train_test)}/{len(y_train_test)}")
            split_scores: dict = {}
            best_score: float = -1
            # train models in model_dict
            for key in tqdm(model_dict.keys(), desc=f"split {split_idx+1}", leave=leave_loadbar):
                # train data classes in first split on all train data
                if split_idx == 0:
                    pre_x, _ = model_dict[key]._Pipeline__data_prepare(x_train, y_train)
                    logger.debug(f"total length of train data after pipeline pre-processing: {len(pre_x)} ({key})")

                # XGBoostClassifier has different warm_start implementation
                if model_dict[key].model_type != "XGBC" or split_idx==0:
                    tscore, ttime = model_dict[key].train_warm_start(x_train_train, y_train_train, scoring=scoring, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength, console_out=False)
                else:
                    start = time.time()
                    model_dict[key].fit_warm_start(x_train_train, y_train_train, xgb_model=model_dict[key].model)
                    end = time.time()
                    tscore, ttime = model_dict[key].evaluate_score(x_train_train, y_train_train, scoring=scoring, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength), str(timedelta(seconds=int(end-start)))
                
                score = model_dict[key].evaluate(x_train_test, y_train_test, avg=avg, pos_label=pos_label, console_out=False, secondary_scoring=secondary_scoring, strength=strength)
                score["train_score"] = tscore
                score["train_time"] = ttime
                split_scores[key] = score
                sorted_split_scores = dict(sorted(split_scores.items(), key=lambda item: (item[1][scoring], item[1]["s_score"], item[1]["train_time"]), reverse=True))
                if score[scoring] > best_score:
                    best_model_name = list(sorted_split_scores.keys())[0]
                    logger.info(f"new best {scoring}: {best_score} -> {score[scoring]} ({best_model_name})")
                    best_score = score[scoring]

            sorted_split_scores_pd = pd.DataFrame(sorted_split_scores).transpose()

            # save model scores
            if save_results_path is not None:
                sorted_split_scores_pd.to_csv(save_results_path.split(".")[0]+f"_split{split_idx+1}."+save_results_path.split(".")[1])

            logger.info(f"Split scores (top 5): \n{sorted_split_scores_pd.head(5)}")

            # only keep better half of the models
            for key in list(sorted_split_scores.keys())[int(len(sorted_split_scores)/2):]:
                del model_dict[key]

            logger.info(f"removed {len(sorted_split_scores)-len(model_dict)} models")
            
            best_model_name = list(sorted_split_scores.keys())[0]
            best_model = model_dict[list(sorted_split_scores.keys())[0]]

        logger.info(f"Evaluating best model: \n\n{best_model_name}\n")
        x_train_train = x_train[int(split_idx*1/split_num*len(x_train)):]
        y_train_train = y_train[int(split_idx*1/split_num*len(y_train)):]
        tscore, ttime = best_model.train_warm_start(x_train_train, y_train_train, console_out=False)
        score = best_model.evaluate(x_test, y_test, avg=avg, pos_label=pos_label, console_out=True, secondary_scoring=secondary_scoring, strength=strength)
        score["train_score"] = tscore
        score["train_time"] = ttime
        return best_model_name, score
