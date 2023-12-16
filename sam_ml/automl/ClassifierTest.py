import os
import sys
import warnings
from typing import Callable, Literal

import pandas as pd

from sam_ml.config import (
    get_avg,
    get_pos_label,
    get_scoring,
    get_secondary_scoring,
    get_strength,
    setup_logger,
)
from sam_ml.data.preprocessing import (
    Embeddings_builder,
    Sampler,
    SamplerPipeline,
    Scaler,
    Selector,
)
from sam_ml.models.classifier.AdaBoostClassifier import ABC
from sam_ml.models.classifier.BaggingClassifier import BC
from sam_ml.models.classifier.BernoulliNB import BNB
from sam_ml.models.classifier.DecisionTreeClassifier import DTC
from sam_ml.models.classifier.ExtraTreesClassifier import ETC
from sam_ml.models.classifier.GaussianNB import GNB
from sam_ml.models.classifier.GaussianProcessClassifier import GPC
from sam_ml.models.classifier.GradientBoostingMachine import GBM
from sam_ml.models.classifier.KNeighborsClassifier import KNC
from sam_ml.models.classifier.LinearDiscriminantAnalysis import LDA
from sam_ml.models.classifier.LinearSupportVectorClassifier import LSVC
from sam_ml.models.classifier.LogisticRegression import LR
from sam_ml.models.classifier.MLPClassifier import MLPC
from sam_ml.models.classifier.QuadraticDiscriminantAnalysis import QDA
from sam_ml.models.classifier.RandomForestClassifier import RFC
from sam_ml.models.classifier.SupportVectorClassifier import SVC
from sam_ml.models.classifier.XGBoostClassifier import XGBC
from sam_ml.models.main_classifier import Classifier

from .main_auto_ml import AutoML

logger = setup_logger(__name__)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affects subprocesses


class CTest(AutoML):
    """ AutoML class for classifier """

    def __init__(self, models: Literal["all", "big_data", "basic", "basic2"] | list[Classifier] = "all", vectorizer: str | Embeddings_builder | None | list[str | Embeddings_builder | None] = None, scaler: str | Scaler | None | list[str | Scaler | None] = None, selector: str | tuple[str, int] | Selector | None | list[str | tuple[str, int] | Selector | None] = None, sampler: str | Sampler | SamplerPipeline | None | list[str | Sampler | SamplerPipeline | None] = None):
        """
        Parameters
        ----------
        models : {"all", "big_data", "basic", "basic2"} or list, default="all"
            - 'all':
                use all Wrapperclass models (18+ models)
            - 'big_data':
                use all Wrapperclass models except the ones that take too much space or 
                time on big data (>200.000 data points)
            - 'basic':
                use basic Wrapperclass models (8 models) which includes:
                LogisticRegression, MLP Classifier, LinearSVC, DecisionTreeClassifier,
                RandomForestClassifier, SVC, Gradientboostingmachine, KNeighborsClassifier
            - 'basic2':
                use basic (mostly tree-based) Wrapperclass models
            - list of Wrapperclass models from sam_ml library

        vectorizer : str, Embeddings_builder, or None
            object or algorithm of :class:`Embeddings_builder` class which will be used for automatic string column vectorizing (None for no vectorizing)
        scaler : str, Scaler, or None
            object or algorithm of :class:`Scaler` class for scaling the data (None for no scaling)
        selector : str, Selector, or None
            object or algorithm of :class:`Selector` class for feature selection (None for no selecting)
        sampling : str, Sampler, or None
            object or algorithm of :class:`Sampler` class for sampling the train data (None for no sampling)

        Notes
        -----
        If a list is provided for one or multiple of the preprocessing steps, all model with preprocessing steps combination will be added as pipelines.
        """
        super().__init__(models, vectorizer, scaler, selector, sampler)

    @staticmethod
    def model_combs(kind: Literal["all", "big_data", "basic", "basic2"]):
        """
        Function for mapping string to set of models

        Parameters
        ----------
        kind : {"all", "big_data", "basic", "basic2"}
            which kind of model set to use:

            - 'all':
                use all Wrapperclass models (18+ models)
            - 'big_data':
                use all Wrapperclass models except the ones that take too much space or 
                time on big data (>200.000 data points)
            - 'basic':
                use basic Wrapperclass models (8 models) which includes:
                LogisticRegression, MLP Classifier, LinearSVC, DecisionTreeClassifier,
                RandomForestClassifier, SVC, Gradientboostingmachine, KNeighborsClassifier
            - 'basic2':
                use basic (mostly tree-based) Wrapperclass models

        Returns
        -------
        models : list
            list of model instances
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
                ABC(estimator="DTC"),
                ABC(estimator="RFC"),
                ABC(estimator="LR"),
                KNC(),
                ETC(),
                GNB(),
                BNB(),
                GPC(),
                BC(estimator="DTC"),
                BC(estimator="RFC"),
                BC(estimator="LR"),
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
        elif kind == "big_data":
            models = [
                LR(),
                QDA(),
                LDA(),
                LSVC(),
                DTC(),
                RFC(),
                GBM(),
                ABC(estimator="DTC"),
                ABC(estimator="RFC"),
                ABC(estimator="LR"),
                ETC(),
                GNB(),
                BNB(),
                BC(estimator="DTC"),
                BC(estimator="RFC"),
                BC(estimator="LR"),
                XGBC(),
            ]
        elif kind == "basic2":
            models = [
                LR(),
                RFC(),
                ABC(estimator="DTC"),
                ABC(estimator="RFC"),
                ABC(estimator="LR"),
                BC(estimator="DTC"),
                BC(estimator="RFC"),
                BC(estimator="LR"),
                XGBC(),
            ]
        else:
            raise ValueError(f"Cannot find model combination '{kind}'")

        return models

    def output_scores_as_pd(self, sort_by: Literal["index", "accuracy", "precision", "recall", "s_score", "l_score", "custom_score", "train_score", "train_time"] | list[str] = "index", console_out: bool = True) -> pd.DataFrame:
        """
        Function to output self.scores as pd.DataFrame

        Parameters
        ----------
        sorted_by : {"index", "accuracy", "precision", "recall", "s_score", "l_score", "custom_score", "train_score", "train_time"} or list[str], default="index"
            key(s) to sort the scores by. You can provide also keys that are not in self.scores and they will be filtered out.

            - "index":
                sort index (``ascending=True``)
            - "accuracy", "precision", "recall", "s_score", "l_score", "custom_score", "train_score", "train_time":
                sort by these columns (``ascending=False``)
            - list with multiple keys (``ascending=False``), e.g., ['precision', 'recall']:
                sort first by 'precision' and then by 'recall'

        console_out : bool, default=True
            shall the DataFrame be printed out

        Returns
        -------
        scores : pd.DataFrame
            sorted DataFrame of self.scores
        """
        return super().output_scores_as_pd(sort_by=sort_by, console_out=console_out)

    def eval_models(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
    ) -> dict[str, dict]:
        """
        Function to train and evaluate every model

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train the models
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate the models
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        avg : {"micro", "macro", "binary", "weighted"} or None, default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")

        Returns
        -------
        scores : dict[str, dict]
            dictionary with scores for every model as dictionary
    
        also saves metrics in self.scores

        Notes
        -----
        if you interrupt the keyboard during the run of eval_models, the interim result will be returned
        """
        return super().eval_models(x_train, y_train, x_test, y_test, scoring=scoring, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)

    def eval_models_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 5,
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        small_data_eval: bool = False,
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        custom_score: Callable[[list[int], list[int]], float] | None = None,
    ) -> dict[str, dict]:
        """
        Function to run a cross validation on every model

        Parameters
        ----------
        X, y : pd.DataFrame, pd.Series
            Data to cross validate on
        cv_num : int, default=5
            number of different random splits (only used when ``small_data_eval=False``)
        avg : {"micro", "macro", "binary", "weighted"} or None, default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        small_data_eval : bool, default=False
            if True, cross_validation_small_data will be used (one-vs-all evaluation). Otherwise, random split cross validation
        custom_score : callable or None, default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict[str, dict]
            dictionary with scores for every model as dictionary
    
        also saves metrics in self.scores

        Notes
        -----
        if you interrupt the keyboard during the run of eval_models_cv, the interim result will be returned
        """
        return super().eval_models_cv(X, y, cv_num=cv_num, small_data_eval=small_data_eval, custom_score=custom_score, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)

    def find_best_model_randomCV(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_trails: int = 5,
        cv_num: int = 3,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        small_data_eval: bool = False,
        leave_loadbar: bool = True,
    ) -> dict[str, dict]:
        """
        Function to run a random cross validation hyperparameter search for every model

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train and optimise the models
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate the models
        n_trails : int, default=5
            max number of parameter sets to test
        cv_num : int, default=3
            number of different random splits (only used when ``small_data_eval=False``)
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        avg : {"micro", "macro", "binary", "weighted"} or None, default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        small_data_eval : bool, default=False
            if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
        leave_loadbar : bool, default=True
            shall the loading bar of the randomCVsearch of each individual model be visible after training (True - load bar will still be visible)

        Returns
        -------
        scores : dict[str, dict]
            dictionary with scores for every model as dictionary
    
        also saves metrics in self.scores

        Notes
        -----
        If you interrupt the keyboard during the run of randomCVsearch of a model, the interim result for this model will be used and the next model starts.
        """
        return super().find_best_model_randomCV(x_train, y_train, x_test, y_test, n_trails=n_trails, cv_num=cv_num, scoring=scoring, small_data_eval=small_data_eval, leave_loadbar=leave_loadbar, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
    
    def find_best_model_smac(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_trails: int = 5,
        cv_num: int = 3,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        small_data_eval: bool = False,
        walltime_limit_per_modeltype: int = 600,
        smac_log_level: int = 30,
    ) -> dict[str, dict]:
        """
        Function to run a Hyperparametertuning with SMAC library HyperparameterOptimizationFacade [can only be used in the sam_ml version with swig] for every model

        The smac_search-method will more "intelligent" search your hyperparameter space than the randomCVsearch and 
        returns the best hyperparameter set. Additionally to the n_trails parameter, it also takes a walltime_limit parameter 
        that defines the maximum time in seconds that the search will take.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train and optimise the models
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate the models
        n_trails : int, default=5
            max number of parameter sets to test for each model
        cv_num : int, default=3
            number of different random splits (only used when ``small_data_eval=False``)
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        avg : {"micro", "macro", "binary", "weighted"} or None, default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        small_data_eval : bool, default=False
            if True: trains model on all datapoints except one and does this for all datapoints (recommended for datasets with less than 150 datapoints)
        walltime_limit_per_modeltype : int, default=600
            the maximum time in seconds that SMAC is allowed to run for each model
        smac_log_level : int, default=30
            10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR, 50 - CRITICAL (SMAC3 library log levels)

        Returns
        -------
        scores : dict[str, dict]
            dictionary with scores for every model as dictionary
    
        also saves metrics in self.scores
        """
        return super().find_best_model_smac(x_train, y_train, x_test, y_test, n_trails=n_trails, cv_num=cv_num, scoring=scoring, small_data_eval=small_data_eval, walltime_limit_per_modeltype=walltime_limit_per_modeltype, smac_log_level=smac_log_level, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
    
    def find_best_model_mass_search(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_trails: int = 10,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        leave_loadbar: bool = True,
        save_results_path: str | None = "find_best_model_mass_search_results.csv",
    ) -> tuple[str, dict[str, float]]:
        """
        Function to run a successive halving hyperparameter search for every model

        It uses the ``warm_start`` parameter of the model and is an own implementation

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train and optimise the models
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate the models
        n_trails : int, default=10
            max number of parameter sets to test for each model
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score), default="accuracy"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        avg : {"micro", "macro", "binary", "weighted"} or None, default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        leave_loadbar : bool, default=True
            shall the loading bar of the model training during the different splits be visible after training (True - load bar will still be visible)
        save_result_path : str or None, default="find_best_model_mass_search_results.csv"
            path to use for saving the results after each step. If ``None`` no results will be saved

        Returns
        -------
        best_model_name : str
            name of the best model in search
        score : dict[str, float]
            scores of the best model
        """
        return super().find_best_model_mass_search(x_train, y_train, x_test, y_test, n_trails=n_trails, scoring=scoring, leave_loadbar=leave_loadbar, save_results_path=save_results_path, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
