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

from ..main_auto_ml import AutoML

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
            object, tuple of algorithm and feature number, or algorithm of :class:`Selector` class for feature selection (None for no selecting)
        sampler : str, Sampler, SamplerPipeline, or None
            object or algorithm of :class:`Sampler` / :class:`SamplerPipeline` class for sampling the train data (None for no sampling)

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

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=3000, n_features=4, n_classes=2, weights=[0.9], random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(y)
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # start modelling
        >>> from sam_ml.models.automl import CTest
        >>> 
        >>> # initialise auot-ml class
        >>> ctest = CTest(models = "all", scaler = "standard", sampler = "ros")
        >>> 
        >>> # start eval_models to evaluate all modeltypes on train-test-data
        >>> ctest.eval_models(x_train,y_train,x_test,y_test, scoring="s_score", avg="binary", pos_label=1, secondary_scoring="precision", strength=3)
        >>> 
        >>> # output and sort results
        >>> score_df = ctest.output_scores_as_pd(sort_by=["s_score", "train_time"])
                                                              accuracy    precision   recall      s_score     l_score     train_time  train_score
        AdaBoostClassifier (RFC based) (vec=None, scale...    0.981667    0.925926    0.877193    0.982953    0.999998    0:00:01     0.994996
        XGBClassifier (vec=None, scaler=standard, selec...    0.981667    0.925926    0.877193    0.982953    0.999998    0:00:00     0.995061
        ExtraTreesClassifier (vec=None, scaler=standard...    0.980000    0.941176    0.842105    0.981653    0.999987    0:00:00     0.995061
        RandomForestClassifier (vec=None, scaler=standa...    0.980000    0.909091    0.877193    0.980948    0.999998    0:00:00     0.995061
        ...
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

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=3000, n_features=4, n_classes=2, weights=[0.9], random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(y)
        >>> 
        >>> # start modelling
        >>> from sam_ml.models.automl import CTest
        >>> 
        >>> # initialise auot-ml class
        >>> ctest = CTest(models = "all", scaler = "standard", sampler = "ros")
        >>> 
        >>> # start eval_models_cv which will cross validate all model types
        >>> ctest.eval_models_cv(X,y, avg="binary", pos_label=1, secondary_scoring="precision", strength=3, cv_num=3)
        >>> 
        >>> # output and sort results
        >>> score_df = ctest.output_scores_as_pd(sort_by=["s_score", "train_time"])
                                                              accuracy    precision   recall      s_score     l_score     train_time  train_score
        AdaBoostClassifier (RFC based) (vec=None, scale...    0.983333    0.943567    0.893781    0.984974    0.999999    0:00:01     1.000000
        XGBClassifier (vec=None, scaler=standard, selec...    0.983667    0.947069    0.894857    0.984930    0.999998    0:00:00     0.999833
        RandomForestClassifier (vec=None, scaler=standa...    0.983000    0.943373    0.891329    0.984505    0.999998    0:00:00     0.999833
        ExtraTreesClassifier (vec=None, scaler=standard...    0.980667    0.932792    0.878323    0.982539    0.999996    0:00:00     1.000000
        ...
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

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=3000, n_features=4, n_classes=2, weights=[0.9], random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(y)
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # start modelling
        >>> from sam_ml.models.automl import CTest
        >>> 
        >>> # initialise auot-ml class
        >>> ctest = CTest(models = "all", scaler = "standard", sampler = "ros")
        >>> 
        >>> # start randomCVsearch with 5 configurations per model type and evaluate the best parameters
        >>> ctest.find_best_model_randomCV(x_train,y_train,x_test,y_test, scoring="s_score", avg="binary", pos_label=1, secondary_scoring="precision", strength=3, n_trails=5, cv_num=3)
        >>> 
        >>> # output and sort results
        >>> score_df = ctest.output_scores_as_pd(sort_by=["s_score", "train_time"])
        randomCVsearch (LogisticRegression (vec=None, scaler=standard, selector=None, sampler=ros)): 100%|██████████| 5/5 [00:00<00:00, 13.74it/s]
        2023-12-08 21:12:57,721 - sam_ml.models.main_auto_ml - INFO - LogisticRegression (vec=None, scaler=standard, selector=None, sampler=ros) - score: 0.8114282933429915 (s_score) - parameters: {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
        <BLANKLINE>
        randomCVsearch (QuadraticDiscriminantAnalysis (vec=None, scaler=standard, selector=None, sampler=ros)): 100%|██████████| 5/5 [00:00<00:00, 19.47it/s]
        2023-12-08 21:12:58,010 - sam_ml.models.main_auto_ml - INFO - QuadraticDiscriminantAnalysis (vec=None, scaler=standard, selector=None, sampler=ros) - score: 0.8788135203591323 (s_score) - parameters: {'reg_param': 0.0}
        <BLANKLINE>
        ...
        <BLANKLINE>
                                                              accuracy    precision   recall      s_score     l_score     train_time  train_score  best_score (rCVs)  best_hyperparameters (rCVs)
        AdaBoostClassifier (DTC based) (vec=None, scale...    0.983333    0.943396    0.877193    0.984656    0.999998    0:00:02     0.995061     0.985320           {'algorithm': 'SAMME', 'estimator__max_depth':...
        AdaBoostClassifier (RFC based) (vec=None, scale...    0.983333    0.943396    0.877193    0.984656    0.999998    0:00:01     0.995061     0.984980           {'algorithm': 'SAMME', 'estimator__max_depth':...
        XGBClassifier (vec=None, scaler=standard, selec...    0.981667    0.942308    0.859649    0.983298    0.999995    0:00:00     0.994929     0.985982           {'colsample_bytree': 1.0, 'gamma': 0.0, 'learn...
        KNeighborsClassifier (vec=None, scaler=standard...    0.980000    0.909091    0.877193    0.980948    0.999998    0:00:00     0.995061     0.978702           {'leaf_size': 37, 'n_neighbors': 2, 'p': 1, 'w...
        ...
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

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=3000, n_features=4, n_classes=2, weights=[0.9], random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(y)
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # start modelling
        >>> from sam_ml.models.automl import CTest
        >>> 
        >>> # initialise auot-ml class
        >>> ctest = CTest(models = "all", scaler = "standard", sampler = "ros")
        >>> 
        >>> # start find_best_model_smac with 5 configurations per model type and evaluate the best parameters
        >>> ctest.find_best_model_smac(x_train,y_train,x_test,y_test, scoring="s_score", avg="binary", pos_label=1, secondary_scoring="precision", strength=3, n_trails=5, cv_num=3)
        >>> 
        >>> # output and sort results
        >>> score_df = ctest.output_scores_as_pd(sort_by=["s_score", "train_time"])
        2023-12-16 14:01:07,878 - sam_ml.automl.main_auto_ml - INFO - LogisticRegression (vec=None, scaler=standard, selector=None, sampler=ros) - parameters: {'C': 1.521010951219768, 'penalty': 'l2', 'solver': 'sag'}
        2023-12-16 14:01:11,110 - sam_ml.automl.main_auto_ml - INFO - QuadraticDiscriminantAnalysis (vec=None, scaler=standard, selector=None, sampler=ros) - parameters: {'reg_param': 0.5455330852419138}
        ...
        <BLANKLINE>
                                                              accuracy    precision   recall      s_score     l_score     train_time  train_score  best_hyperparameters (rCVs)
        GradientBoostingMachine (vec=None, scaler=stand...    0.986667    0.962264    0.894737    0.987177    0.999999    0:00:00     0.995061     {'criterion': 'squared_error', 'learning_rate'...
        AdaBoostClassifier (RFC based) (vec=None, scale...    0.983333    0.943396    0.877193    0.984656    0.999998    0:00:06     0.995061     {'algorithm': 'SAMME', 'estimator__max_depth':...
        MLP Classifier (vec=None, scaler=standard, sele...    0.981667    0.883333    0.929825    0.979544    0.999996    0:00:04     0.989968     {'activation': 'relu', 'alpha': 0.031488905303...
        AdaBoostClassifier (DTC based) (vec=None, scale...    0.978333    0.958333    0.807018    0.978841    0.999937    0:00:00     0.995061     {'algorithm': 'SAMME', 'estimator__max_depth':...
        ...
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

        It uses the ``warm_start`` parameter of the model and is an own implementation.
        Recommended to use as a fast method to narrow down different preprocessing steps and model combinations, but ``find_best_model_smac`` or ``randomCVsearch`` return better results.

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

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=3000, n_features=4, n_classes=2, weights=[0.9], random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(y)
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # start modelling
        >>> from sam_ml.models.automl import CTest
        >>> 
        >>> # initialise auot-ml class
        >>> ctest = CTest(models = "all", scaler = "standard", sampler = "ros")
        >>> 
        >>> # start find_best_model_mass_search with 10 configurations per model type and find the best combination
        >>> best_model_name, score = ctest.find_best_model_mass_search(x_train,y_train,x_test,y_test, scoring="s_score", avg="binary", pos_label=1, secondary_scoring="precision", strength=3, n_trails=10)
        2023-12-16 14:33:18,352 - sam_ml.automl.main_auto_ml - WARNING - modeltype in 'QuadraticDiscriminantAnalysis (vec=None, scaler=standard, selector=None, sampler=ros)' is not supported for this search -> will be skipped
        2023-12-16 14:33:18,354 - sam_ml.automl.main_auto_ml - WARNING - modeltype in 'LinearDiscriminantAnalysis (vec=None, scaler=standard, selector=None, sampler=ros)' is not supported for this search -> will be skipped
        ...
        2023-12-16 14:33:18,493 - sam_ml.automl.main_auto_ml - INFO - total number of models: 99
        2023-12-16 14:33:18,493 - sam_ml.automl.main_auto_ml - INFO - split number: 6, split_size (x_train): 342
        2023-12-16 14:33:18,500 - sam_ml.automl.main_auto_ml - INFO - split 1: length x_train/y_train 342/342, length x_test/y_test 2058/2058
        2023-12-16 14:33:18,548 - sam_ml.automl.main_auto_ml - INFO - new best s_score: -1 -> 0.9693235800399905 (LogisticRegression (vec=None, scaler=standard, selector=None, sampler=ros) {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'})
        2023-12-16 14:33:18,574 - sam_ml.automl.main_auto_ml - INFO - new best s_score: 0.9693235800399905 -> 0.9728086992300002 (LogisticRegression (vec=None, scaler=standard, selector=None, sampler=ros) {'C': 63.512210106407046, 'penalty': 'l2', 'solver': 'lbfgs'})
        ...
        2023-12-16 14:33:53,863 - sam_ml.automl.main_auto_ml - INFO - Split scores (top 5):
                                                              accuracy    precision   recall      s_score     l_score     train_time  train_score
        BaggingClassifier (DTC based) (vec=None, scaler...    0.97862     0.956098    0.848485    0.983412    0.99999     0:00:00     0.97371
        GaussianProcessClassifier (vec=None, scaler=sta...    0.977648    0.938389    0.857143    0.982668    0.999994    0:00:00     0.971741
        ...
        2023-12-16 14:39:43,242 - sam_ml.automl.main_auto_ml - INFO - removed 50 models
        2023-12-16 14:39:43,243 - sam_ml.automl.main_auto_ml - INFO - split 2: length x_train/y_train 342/342, length x_test/y_test 1716/1716
        2023-12-16 14:39:43,275 - sam_ml.automl.main_auto_ml - INFO - new best s_score: -1 -> 0.9711723626069725 (LogisticRegression (vec=None, scaler=standard, selector=None, sampler=ros) {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'})
        2023-12-16 14:39:43,299 - sam_ml.automl.main_auto_ml - INFO - new best s_score: 0.9711723626069725 -> 0.9720794877353701 (LogisticRegression (vec=None, scaler=standard, selector=None, sampler=ros) {'C': 63.512210106407046, 'penalty': 'l2', 'solver': 'lbfgs'})
        ...
        2023-12-16 14:39:56,237 - sam_ml.automl.main_auto_ml - INFO - removed 25 models
        2023-12-16 14:39:56,238 - sam_ml.automl.main_auto_ml - INFO - split 3: length x_train/y_train 342/342, length x_test/y_test 1374/1374
        ...
        2023-12-16 14:39:59,039 - sam_ml.automl.main_auto_ml - INFO - removed 12 models
        2023-12-16 14:39:59,041 - sam_ml.automl.main_auto_ml - INFO - split 4: length x_train/y_train 342/342, length x_test/y_test 1032/1032
        ...
        2023-12-16 14:40:00,704 - sam_ml.automl.main_auto_ml - INFO - removed 6 models
        2023-12-16 14:40:00,705 - sam_ml.automl.main_auto_ml - INFO - split 5: length x_train/y_train 342/342, length x_test/y_test 690/690
        ...
        2023-12-16 14:40:02,124 - sam_ml.automl.main_auto_ml - INFO - removed 3 models
        2023-12-16 14:40:02,126 - sam_ml.automl.main_auto_ml - INFO - split 6: length x_train/y_train 342/342, length x_test/y_test 348/348
        ...
        2023-12-16 14:40:02,369 - sam_ml.automl.main_auto_ml - INFO - removed 2 models
        2023-12-16 14:40:02,370 - sam_ml.automl.main_auto_ml - INFO - Evaluating best model:
        <BLANKLINE>
        XGBClassifier (vec=None, scaler=standard, selector=None, sampler=ros) {'colsample_bytree': 1.0, 'gamma': 0.0, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 1.0}
        <BLANKLINE>
        accuracy: 0.9783333333333334
        precision: 0.94
        recall: 0.8245614035087719
        s_score: 0.9796565264661882
        l_score: 0.9999699295994755
        <BLANKLINE>
        classification report:
                        precision   recall  f1-score    support
        <BLANKLINE>
                0       0.98        0.99    0.99        543
                1       0.94        0.82    0.88        57
        <BLANKLINE>
        accuracy                            0.98        600
        macro avg       0.96        0.91    0.93        600
        weighted avg    0.98        0.98    0.98        600
        <BLANKLINE>
        """
        return super().find_best_model_mass_search(x_train, y_train, x_test, y_test, n_trails=n_trails, scoring=scoring, leave_loadbar=leave_loadbar, save_results_path=save_results_path, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
