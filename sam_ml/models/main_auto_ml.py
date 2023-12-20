import os
import sys
import time
import warnings
from abc import abstractmethod
from datetime import timedelta
from inspect import isfunction
from typing import Callable

import numpy as np
import pandas as pd

# to deactivate pygame promt 
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
from pkg_resources import resource_filename
from tqdm.auto import tqdm

from sam_ml.config import get_sound_on, setup_logger
from sam_ml.data.preprocessing import (
    Embeddings_builder,
    Sampler,
    SamplerPipeline,
    Scaler,
    Selector,
)
from sam_ml.models.main_pipeline import create_pipeline

logger = setup_logger(__name__)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affects subprocesses


class AutoML:
    """ Auto-ML parent class {abstract} """

    def __init__(self, models: str | list, vectorizer: str | Embeddings_builder | None | list[str | Embeddings_builder | None], scaler: str | Scaler | None | list[str | Scaler | None], selector: str | tuple[str, int] | Selector | None | list[str | tuple[str, int] | Selector | None], sampler: str | Sampler | SamplerPipeline | None  | list[str | Sampler | SamplerPipeline | None]):
        """
        Parameters
        ----------
        models : str or list

            - string of model set from ``model_combs`` method
            - list of Wrapperclass models from sam_ml library

        vectorizer : str, Embeddings_builder, or None
            object or algorithm of :class:`Embeddings_builder` class which will be used for automatic string column vectorizing (None for no vectorizing)
        scaler : str, Scaler, or None
            object or algorithm of :class:`Scaler` class for scaling the data (None for no scaling)
        selector : str, Selector, or None
            object, tuple of algorithm and feature number, or algorithm of :class:`Selector` class for feature selection (None for no selecting)
        sampler : str, Sampler, SamplerPipeline, or None
            object or algorithm of :class:`Sampler` / :class:`SamplerPipeline` class for sampling the train data (None for no sampling)

        Notes
        -----
        If a list is provided for one or multiple of the preprocessing steps, all model with preprocessing steps combination will be added as pipelines
        """
        self.__models_input = models

        if type(models) == str:
            models = self.model_combs(models)

        if type(vectorizer) in (str, Embeddings_builder) or vectorizer is None:
            vectorizer = [vectorizer]

        if type(scaler) in (str, Scaler) or scaler is None:
            scaler = [scaler]

        if type(selector) in (str, tuple, Selector) or selector is None:
            selector = [selector]

        if type(sampler) in (str, Sampler) or sampler is None:
            sampler = [sampler]

        self._vectorizer = vectorizer
        self._scaler = scaler
        self._selector = selector
        self._sampler = sampler

        self._models: dict = {}
        for model in models:
            self.add_model(model)

        self._scores: dict = {}

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

        return f"{self.__class__.__name__}({params})"
    
    @property
    def scores(self) -> dict[str, float]:
        """
        Returns
        -------
        scores : dict[str, float]
            dictionary with scores for every model as dictionary
        """
        return self._scores
    
    @property
    def models(self) -> dict:
        """
        Returns
        -------
        models : dict
            dictionary with model names as keys and model instances as values
        """
        return self._models

    def remove_model(self, model_name: str):
        """
        Function for deleting model in self.models

        Parameters
        ----------
        model_name : str
            name of model in self.models
        """
        del self._models[model_name]

    def add_model(self, model):
        """
        Function for adding model in self.models

        Parameters
        ----------
        model : estimator instance
            add model instance to self.models
        """
        for vec in self._vectorizer:
            for scal in self._scaler:
                for sel in self._selector:
                    for sam in self._sampler:
                        model_pipe_name = model.model_name+f" (vec={vec}, scaler={scal}, selector={sel}, sampler={sam})"
                        self._models[model_pipe_name] = create_pipeline(vectorizer=vec,  scaler=scal, selector=sel, sampler=sam, model=model, model_name=model_pipe_name)

    @staticmethod
    @abstractmethod
    def model_combs(kind: str) -> list:
        """
        Function for mapping string to set of models

        Parameters
        ----------
        kind : str
            which kind of model set to use:

            - "all": 
                use all models
            - ...

        Returns
        -------
        models : list
            list of model instances
        """
        pass

    @staticmethod
    def __finish_sound():
        """ little function to play a microwave sound """
        if get_sound_on():
            filepath = resource_filename(__name__, 'microwave_finish_sound.mp3')
            pygame.mixer.init()
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()

    @staticmethod
    def __sort_dict(scores: dict, sort_by: list[str]) -> pd.DataFrame:
        """
        Function to sort a dict by a given list of keys

        Parameters
        ----------
        scores : dict
            dictionary with scores
        sorted_by : list[str]
            keys to sort the scores by. You can provide also keys that are not in scores and they will be filtered out.

        Returns
        -------
        scores_df : pd.DataFrame
            sorted dataframe of scores
        """
        scores_df = pd.DataFrame.from_dict(scores, orient="index")

        # Filter sort_by to include only those present in the DataFrame
        existing_sort_columns = [col for col in sort_by if col in scores_df.columns]
        # Sort the DataFrame by the existing columns, if there are any
        if existing_sort_columns:
            scores_df = scores_df.sort_values(by=existing_sort_columns, ascending=False)

        return scores_df

    def output_scores_as_pd(self, sort_by: str | list[str], console_out: bool) -> pd.DataFrame:
        """
        Function to output self.scores as pd.DataFrame

        Parameters
        ----------
        sorted_by : str or list[str]
            key(s) to sort the scores by. You can provide also keys that are not in self.scores and they will be filtered out.
        console_out : bool
            shall the DataFrame be printed out

        Returns
        -------
        scores : pd.DataFrame
            sorted DataFrame of self.scores
        """
        if self._scores != {}:
            if sort_by == "index":
                scores = pd.DataFrame.from_dict(self._scores, orient="index").sort_index(ascending=True)
            else:
                scores = self.__sort_dict(self._scores, sort_by=sort_by)

            if console_out:
                print(scores)
        else:
            logger.warning("no scores are created -> use a method that saves it scores in self.scores")
            scores = None

        return scores

    def eval_models(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: str | Callable,
        **kwargs,
    ) -> dict[str, dict]:
        """
        Function to train and evaluate every model

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train the models
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate the models
        scoring : str or callable (custom score)
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`
        \*\*kwargs:
            additional parameters from child-class for ``evaluate`` method of models

        Returns
        -------
        scores : dict[str, dict]
            dictionary with scores for every model as dictionary
    
        also saves metrics in self.scores

        Notes
        -----
        if you interrupt the keyboard during the run of eval_models, the interim result will be returned
        """
        if isfunction(scoring):
            custom_score = scoring
        else:
            custom_score = None

        try:
            for key in tqdm(self._models.keys(), desc="Evaluation"):
                tscore, ttime = self._models[key].train(x_train, y_train, console_out=False, scoring=scoring)
                score = self._models[key].evaluate(
                    x_test, y_test, console_out=False, **kwargs, custom_score=custom_score,
                )
                score["train_time"] = ttime
                score["train_score"] = tscore
                self._scores[key] = score

            self.__finish_sound()

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt - output interim result")
            
        return self._scores

    def eval_models_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int,
        small_data_eval: bool,
        custom_score: Callable | None,
        **kwargs,
    ) -> dict[str, dict]:
        """
        Function to run a cross validation on every model

        Parameters
        ----------
        X, y : pd.DataFrame, pd.Series
            Data to cross validate on
        cv_num : int
            number of different random splits (only used when ``small_data_eval=False``)
        small_data_eval : bool
            if True, cross_validation_small_data will be used (one-vs-all evaluation). Otherwise, random split cross validation
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.
        \*\*kwargs:
            additional parameters from child-class for ``cross validation`` methods of models

        Returns
        -------
        scores : dict[str, dict]
            dictionary with scores for every model as dictionary
    
        also saves metrics in self.scores

        Notes
        -----
        if you interrupt the keyboard during the run of eval_models_cv, the interim result will be returned
        """
        try:
            for key in tqdm(self._models.keys(), desc="Crossvalidation"):
                if small_data_eval:
                    self._models[key].cross_validation_small_data(
                        X, y, console_out=False, leave_loadbar=False, **kwargs, custom_score=custom_score,
                    )
                else:
                    self._models[key].cross_validation(
                        X, y, cv_num=cv_num, console_out=False, **kwargs, custom_score=custom_score,
                    )
                self._scores[key] = self._models[key].cv_scores
            self.__finish_sound()

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt - output interim result")
        
        return self._scores

    def find_best_model_randomCV(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_trails: int,
        cv_num: int,
        scoring: str | Callable,
        small_data_eval: bool,
        leave_loadbar: bool,
        **kwargs,
    ) -> dict[str, dict]:
        """
        Function to run a random cross validation hyperparameter search for every model

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train and optimise the models
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate the models
        n_trails : int
            max number of parameter sets to test
        cv_num : int
            number of different random splits (only used when ``small_data_eval=False``)
        scoring : str or callable (custom score)
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`
        small_data_eval : bool
            if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
        leave_loadbar : bool
            shall the loading bar of the randomCVsearch of each individual model be visible after training (True - load bar will still be visible)
        \*\*kwargs:
            additional parameters from child-class for ``randomCVsearch`` and ``evaluate`` method of models

        Returns
        -------
        scores : dict[str, dict]
            dictionary with scores for every model as dictionary
    
        also saves metrics in self.scores

        Notes
        -----
        If you interrupt the keyboard during the run of randomCVsearch of a model, the interim result for this model will be used and the next model starts.
        """
        if isfunction(scoring):
            custom_score = scoring
        else:
            custom_score = None

        for key in tqdm(self._models.keys(), desc="randomCVsearch"):
            best_hyperparameters, best_score = self._models[key].randomCVsearch(x_train, y_train, n_trails=n_trails, scoring=scoring, small_data_eval=small_data_eval, cv_num=cv_num, leave_loadbar=leave_loadbar, **kwargs)
            if isfunction(scoring):
                logger.info(f"{self._models[key].model_name} - score: {best_score} (custom_score) - parameters: {best_hyperparameters}")
            else:
                logger.info(f"{self._models[key].model_name} - score: {best_score} ({scoring}) - parameters: {best_hyperparameters}")
            if best_hyperparameters:
                model_best = self._models[key].get_deepcopy()
                model_best.set_params(**best_hyperparameters)
                train_score, train_time = model_best.train(x_train, y_train, console_out=False, scoring=scoring)
                scores = model_best.evaluate(x_test, y_test, console_out=False, custom_score=custom_score, **kwargs)
                
                scores["train_time"] = train_time
                scores["train_score"] = train_score
                scores["best_score (rCVs)"] = best_score
                scores["best_hyperparameters (rCVs)"] = best_hyperparameters
                self._scores[key] = scores

        if isfunction(scoring):
            scoring = "custom_score"
        
        sorted_scores = self.output_scores_as_pd(sort_by=[scoring, "s_score", "r2", "train_time"], console_out=False)
        best_model_type = sorted_scores.iloc[0].name
        best_model_value = sorted_scores.iloc[0][scoring]
        best_model_hyperparameters = sorted_scores.iloc[0]["best_hyperparameters (rCVs)"]
        logger.info(f"best model type {best_model_type} - {scoring}: {best_model_value} - parameters: {best_model_hyperparameters}")
        self.__finish_sound()
        return self._scores
    
    def find_best_model_smac(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_trails: int,
        cv_num: int,
        scoring: str | Callable,
        small_data_eval: bool,
        walltime_limit_per_modeltype: int,
        smac_log_level: int,
        **kwargs,
    ) -> dict[str, dict]:
        """
        Function to run a Hyperparametertuning with SMAC library HyperparameterOptimizationFacade for every model [can only be used in the sam_ml version with swig]

        The smac_search-method will more "intelligent" search your hyperparameter space than the randomCVsearch and 
        returns the best hyperparameter set. Additionally to the n_trails parameter, it also takes a walltime_limit parameter 
        that defines the maximum time in seconds that the search will take.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train and optimise the models
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate the models
        n_trails : int
            max number of parameter sets to test for each model
        cv_num : int
            number of different random splits (only used when ``small_data_eval=False``)
        scoring : str or callable (custom score)
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`
        small_data_eval : bool
            if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
        walltime_limit_per_modeltype : int
            the maximum time in seconds that SMAC is allowed to run for each model
        smac_log_level : int
            10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR, 50 - CRITICAL (SMAC3 library log levels)
        \*\*kwargs:
            additional parameters from child-class for ``smac_search`` and ``evaluate`` method of models

        Returns
        -------
        scores : dict[str, dict]
            dictionary with scores for every model as dictionary
    
        also saves metrics in self.scores
        """
        if isfunction(scoring):
            custom_score = scoring
        else:
            custom_score = None

        for key in tqdm(self._models.keys(), desc="smac_search"):
            best_hyperparameters = self._models[key].smac_search(x_train, y_train, n_trails=n_trails, scoring=scoring, small_data_eval=small_data_eval, cv_num=cv_num, walltime_limit=walltime_limit_per_modeltype, log_level=smac_log_level, **kwargs)
            logger.info(f"{self._models[key].model_name} - parameters: {dict(best_hyperparameters)}")
            
            model_best = self._models[key].get_deepcopy()
            model_best.set_params(**best_hyperparameters)
            train_score, train_time = model_best.train(x_train, y_train, console_out=False, scoring=scoring)
            scores = model_best.evaluate(x_test, y_test, console_out=False, custom_score=custom_score, **kwargs)
            
            scores["train_time"] = train_time
            scores["train_score"] = train_score
            scores["best_hyperparameters"] = dict(best_hyperparameters)
            self._scores[key] = scores

        if isfunction(scoring):
            scoring = "custom_score"
        
        sorted_scores = self.output_scores_as_pd(sort_by=[scoring, "s_score", "r2", "train_time"], console_out=False)
        best_model_type = sorted_scores.iloc[0].name
        best_model_value = sorted_scores.iloc[0][scoring]
        best_model_hyperparameters = sorted_scores.iloc[0]["best_hyperparameters"]
        logger.info(f"best model type {best_model_type} - {scoring}: {best_model_value} - parameters: {best_model_hyperparameters}")
        self.__finish_sound()
        return self._scores
    
    def find_best_model_mass_search(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_trails: int,
        scoring: str | Callable,
        leave_loadbar: bool,
        save_results_path: str | None,
        **kwargs,
    ) -> tuple[str, dict[str, float]]:
        """
        Function to run a successive halving hyperparameter search for every model

        It uses the ``warm_start`` parameter of the model and is an own implementation.
        Recommended to use as a fast method to narrow down different preprocessing steps and model combinations, but ``find_best_model_smac`` or ``randomCVsearch`` return better results.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train and optimise the models
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate the models
        n_trails : int
            max number of parameter sets to test for each model
        scoring : str or callable (custom score)
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, \*\*kwargs)`
        leave_loadbar : bool
            shall the loading bar of the model training during the different splits be visible after training (True - load bar will still be visible)
        save_result_path : str or None
            path to use for saving the results after each step. If ``None`` no results will be saved
        \*\*kwargs:
            additional parameters from child-class for ``train_warm_start``, ``evaluate``, and ``evaluate_score`` method of models

        Returns
        -------
        best_model_name : str
            name of the best model in search
        score : dict[str, float]
            scores of the best model
        """
        # create all models
        model_dict = {}
        for key in self._models.keys():
            model = self._models[key]
            configs = model.get_random_configs(n_trails)
            try:
                for config in configs:
                    model_new = model.get_deepcopy()
                    model_new = model_new.set_params(**config)
                    if not model_new.model_type in ["XGBC"] + ["XGBR"]:
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

        # custom score
        if isfunction(scoring):
            custom_score = scoring
            scoring_name = "custom_score"
        else:
            custom_score = None
            scoring_name = scoring

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
                    pre_x, _ = model_dict[key]._data_prepare(x_train, y_train)
                    logger.debug(f"total length of train data after pipeline pre-processing: {len(pre_x)} ({key})")

                # XGBoostClassifier has different warm_start implementation
                if (not model_dict[key].model_type in ["XGBC"] + ["XGBR"]) or split_idx==0:
                    tscore, ttime = model_dict[key].train_warm_start(x_train_train, y_train_train, scoring=scoring, console_out=False, **kwargs)
                else:
                    start = time.time()
                    model_dict[key].fit_warm_start(x_train_train, y_train_train, xgb_model=model_dict[key].model)
                    end = time.time()
                    tscore, ttime = model_dict[key].evaluate_score(x_train_train, y_train_train, scoring=scoring, **kwargs), str(timedelta(seconds=int(end-start)))

                score = model_dict[key].evaluate(x_train_test, y_train_test, console_out=False, custom_score=custom_score, **kwargs)
                score["train_time"] = ttime
                score["train_score"] = tscore
                split_scores[key] = score


                sorted_split_scores = self.__sort_dict(split_scores, sort_by=[scoring_name, "s_score", "r2", "train_time"]).transpose().to_dict()
                if score[scoring_name] > best_score:
                    best_model_name = list(sorted_split_scores.keys())[0]
                    logger.info(f"new best {scoring_name}: {best_score} -> {score[scoring_name]} ({best_model_name})")
                    best_score = score[scoring_name]

            sorted_split_scores_pd = pd.DataFrame(sorted_split_scores).transpose()

            # save model scores
            if save_results_path is not None:
                sorted_split_scores_pd.to_csv(save_results_path.split(".")[0]+f"_split{split_idx+1}."+save_results_path.split(".")[1])

            logger.info(f"Split scores (top {len(sorted_split_scores_pd.head(5))}): \n{sorted_split_scores_pd.head(5)}")

            # only keep better half of the models
            for key in list(sorted_split_scores.keys())[int(len(sorted_split_scores)/2):]:
                model_dict.pop(key)

            logger.info(f"removed {len(sorted_split_scores)-len(model_dict)} models")
            
            best_model_name = list(sorted_split_scores.keys())[0]
            best_model = model_dict[list(sorted_split_scores.keys())[0]]

        logger.info(f"Evaluating best model: \n\n{best_model_name}\n")
        x_train_train = x_train[int(split_idx*1/split_num*len(x_train)):]
        y_train_train = y_train[int(split_idx*1/split_num*len(y_train)):]
        tscore, ttime = best_model.train_warm_start(x_train_train, y_train_train, console_out=False, scoring=scoring)
        score = best_model.evaluate(x_test, y_test, console_out=True, custom_score=custom_score, **kwargs)
        score["train_time"] = ttime
        score["train_score"] = tscore
        return best_model_name, score
