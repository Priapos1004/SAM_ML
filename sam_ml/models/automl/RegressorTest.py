import os
import sys
import warnings
from typing import Callable, Literal

import pandas as pd

# to deactivate pygame promt 
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from sam_ml.config import setup_logger
from sam_ml.data.preprocessing import (
    Embeddings_builder,
    Sampler,
    SamplerPipeline,
    Scaler,
    Selector,
)
from sam_ml.models.main_regressor import Regressor
from sam_ml.models.regressor.BayesianRidge import BYR
from sam_ml.models.regressor.DecisionTreeRegressor import DTR
from sam_ml.models.regressor.ElasticNet import EN
from sam_ml.models.regressor.ExtraTreesRegressor import ETR
from sam_ml.models.regressor.LassoLarsCV import LLCV
from sam_ml.models.regressor.RandomForestRegressor import RFR
from sam_ml.models.regressor.SGDRegressor import SGDR
from sam_ml.models.regressor.XGBoostRegressor import XGBR

from ..main_auto_ml import AutoML

logger = setup_logger(__name__)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affects subprocesses


class RTest(AutoML):
    """ AutoML class for regressor """

    def __init__(self, models: Literal["all"] | list[Regressor] = "all", vectorizer: str | Embeddings_builder | None | list[str | Embeddings_builder | None] = None, scaler: str | Scaler | None | list[str | Scaler | None] = None, selector: str | tuple[str, int] | Selector | None | list[str | tuple[str, int] | Selector | None] = None, sampler: str | Sampler | SamplerPipeline | None | list[str | Sampler | SamplerPipeline | None] = None):
        """
        Parameters
        ----------
        models : {"all"} or list, default="all"
            - 'all':
                use all Wrapperclass models
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

    def model_combs(self, kind: Literal["all"]):
        """
        Function for mapping string to set of models

        Parameters
        ----------
        kind : {"all"}
            which kind of model set to use:

            - 'all':
                use all Wrapperclass models

        Returns
        -------
        models : list
            list of model instances
        """
        if kind == "all":
            models = [
                RFR(),
                DTR(),
                ETR(),
                SGDR(),
                LLCV(),
                EN(),
                BYR(),
                XGBR(),
            ]
        else:
            raise ValueError(f"Cannot find model combination '{kind}'")

        return models

    def output_scores_as_pd(self, sort_by: Literal["index", "r2", "rmse", "d2_tweedie", "custom_score", "train_score", "train_time"] | list[str] = "index", console_out: bool = True) -> pd.DataFrame:
        """
        Function to output self.scores as pd.DataFrame

        Parameters
        ----------
        sorted_by : {"index", "r2", "rmse", "d2_tweedie", "custom_score", "train_score", "train_time"} or list[str], default="index"
            key(s) to sort the scores by. You can provide also keys that are not in self.scores and they will be filtered out.

            - "index":
                sort index (``ascending=True``)
            - "index", "r2", "rmse", "d2_tweedie", "custom_score", "train_score", "train_time":
                sort by these columns (``ascending=False``)
            - list with multiple keys (``ascending=False``), e.g., ['r2', 'd2_tweedie']:
                sort first by 'r2' and then by 'd2_tweedie'

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
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[float], list[float]], float] = "r2",
    ) -> dict[str, dict]:
        """
        Function to train and evaluate every model

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train the models
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate the models
        scoring : {"r2", "rmse", "d2_tweedie"} or callable (custom score), default="r2"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

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
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # start modelling
        >>> from sam_ml.models.automl import RTest
        >>> 
        >>> # initialise auot-ml class
        >>> rtest = RTest(models = "all", scaler = "standard")
        >>> 
        >>> # start eval_models to evaluate all modeltypes on train-test-data
        >>> rtest.eval_models(x_train,y_train,x_test,y_test, scoring="r2")
        >>> 
        >>> # output and sort results
        >>> score_df = rtest.output_scores_as_pd(sort_by=["r2", "train_time"])
                                                              r2          rmse         d2_tweedie   train_time  train_score
        ExtraTreesRegressor (vec=None, scaler=standard,...    0.973874    8.065976     0.954158     0:00:00     1.000000
        RandomForestRegressor (vec=None, scaler=standar...    0.947177    11.469144    0.921423     0:00:00     0.992563
        XGBRegressor (vec=None, scaler=standard, select...    0.945507    11.648970    -1.000000    0:00:00     0.995029
        DecisionTreeRegressor (vec=None, scaler=standar...    0.872008    17.852977    0.788164     0:00:00     1.000000
        ...
        """
        return super().eval_models(x_train, y_train, x_test, y_test, scoring=scoring)

    def eval_models_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 5,
        small_data_eval: bool = False,
        custom_score: Callable[[list[float], list[float]], float] | None = None,
    ) -> dict[str, dict]:
        """
        Function to run a cross validation on every model

        Parameters
        ----------
        X, y : pd.DataFrame, pd.Series
            Data to cross validate on
        cv_num : int, default=5
            number of different random splits (only used when ``small_data_eval=False``)
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
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>> 
        >>> # start modelling
        >>> from sam_ml.models.automl import RTest
        >>> 
        >>> # initialise auot-ml class
        >>> rtest = RTest(models = "all", scaler = "standard")
        >>> 
        >>> # start eval_models_cv which will cross validate all model types
        >>> rtest.eval_models_cv(X,y, cv_num=3)
        >>> 
        >>> # output and sort results
        >>> score_df = rtest.output_scores_as_pd(sort_by=["r2", "train_time"])
                                                              r2          rmse         d2_tweedie  train_time  train_score
        ExtraTreesRegressor (vec=None, scaler=standard,...    0.968637    8.620645     0.947802    0:00:00     1.000000
        XGBRegressor (vec=None, scaler=standard, select...    0.940348    11.946437    0.905749    0:00:00     0.996965
        RandomForestRegressor (vec=None, scaler=standar...    0.939039    12.045418    0.915000    0:00:00     0.992059
        DecisionTreeRegressor (vec=None, scaler=standar...    0.834557    19.894214    0.752740    0:00:00     1.000000
        ...
        """
        return super().eval_models_cv(X, y, cv_num=cv_num, small_data_eval=small_data_eval, custom_score=custom_score)

    def find_best_model_randomCV(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_trails: int = 5,
        cv_num: int = 3,
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[float], list[float]], float] = "r2",
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
        scoring : {"r2", "rmse", "d2_tweedie"} or callable (custom score), default="r2"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
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
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # start modelling
        >>> from sam_ml.models.automl import RTest
        >>> 
        >>> # initialise auot-ml class
        >>> rtest = RTest(models = "all", scaler = "standard")
        >>> 
        >>> # start randomCVsearch with 5 configurations per model type and evaluate the best parameters
        >>> rtest.find_best_model_randomCV(x_train,y_train,x_test,y_test, scoring="r2", n_trails=5, cv_num=3)
        >>> 
        >>> # output and sort results
        >>> score_df = rtest.output_scores_as_pd(sort_by=["r2", "train_time"])
        randomCVsearch (RandomForestRegressor (vec=None, scaler=standard, selector=None, sampler=None)): 100%|██████████| 5/5 [00:01<00:00,  2.89it/s]
        2023-12-18 11:20:25,284 - sam_ml.automl.main_auto_ml - INFO - RandomForestRegressor (vec=None, scaler=standard, selector=None, sampler=None) - score: 0.5568605014361813 (r2) - parameters: {'bootstrap': True, 'criterion': 'squared_error', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100}
        <BLANKLINE>
        randomCVsearch (DecisionTreeRegressor (vec=None, scaler=standard, selector=None, sampler=None)): 100%|██████████| 5/5 [00:06<00:00,  1.30s/it]
        2023-12-18 11:20:31,959 - sam_ml.automl.main_auto_ml - INFO - DecisionTreeRegressor (vec=None, scaler=standard, selector=None, sampler=None) - score: 0.38132412930754 (r2) - parameters: {'criterion': 'squared_error', 'max_depth': 5, 'max_features': 1.0, 'max_leaf_nodes': 90, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'splitter': 'best'}
        <BLANKLINE>
        ...
        <BLANKLINE>
                                                              r2          rmse         d2_tweedie  train_time  train_score   best_score (rCVs)  best_hyperparameters (rCVs)
        XGBRegressor (vec=None, scaler=standard, select...    0.949305    11.235687    0.903326    0:00:00     0.983904      0.930262           {'colsample_bytree': 1.0, 'gamma': 0.0, 'learn...
        RandomForestRegressor (vec=None, scaler=standar...    0.558927    33.141604    0.466038    0:00:00     0.634854      0.556861           {'bootstrap': True, 'criterion': 'squared_erro...
        ExtraTreesRegressor (vec=None, scaler=standard,...    0.473561    36.207000    0.412317    0:00:00     0.513346      0.486602           {'bootstrap': False, 'criterion': 'squared_err...
        DecisionTreeRegressor (vec=None, scaler=standar...    0.431751    37.617306    0.345249    0:00:00     0.522676      0.381324           {'criterion': 'squared_error', 'max_depth': 5,...
        ...
        """
        return super().find_best_model_randomCV(x_train, y_train, x_test, y_test, n_trails=n_trails, cv_num=cv_num, scoring=scoring, small_data_eval=small_data_eval, leave_loadbar=leave_loadbar)
    
    def find_best_model_smac(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_trails: int = 5,
        cv_num: int = 3,
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[float], list[float]], float] = "r2",
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
        scoring : {"r2", "rmse", "d2_tweedie"} or callable (custom score), default="r2"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
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
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # start modelling
        >>> from sam_ml.models.automl import RTest
        >>> 
        >>> # initialise auot-ml class
        >>> rtest = RTest(models = "all", scaler = "standard")
        >>> 
        >>> # start find_best_model_smac with 5 configurations per model type and evaluate the best parameters
        >>> rtest.find_best_model_smac(x_train,y_train,x_test,y_test, scoring="r2", n_trails=5, cv_num=3)
        >>> 
        >>> # output and sort results
        >>> score_df = rtest.output_scores_as_pd(sort_by=["r2", "train_time"])
        2023-12-18 11:32:01,189 - sam_ml.automl.main_auto_ml - INFO - RandomForestRegressor (vec=None, scaler=standard, selector=None, sampler=None) - parameters: {'bootstrap': False, 'criterion': 'friedman_mse', 'max_depth': 13, 'min_samples_leaf': 4, 'min_samples_split': 8, 'min_weight_fraction_leaf': 0.06446314882742665, 'n_estimators': 20}
        2023-12-18 11:32:04,663 - sam_ml.automl.main_auto_ml - INFO - DecisionTreeRegressor (vec=None, scaler=standard, selector=None, sampler=None) - parameters: {'criterion': 'squared_error', 'max_depth': 12, 'max_features': 'sqrt', 'max_leaf_nodes': 43, 'min_samples_leaf': 4, 'min_samples_split': 8, 'min_weight_fraction_leaf': 0.051022405374014035, 'splitter': 'best'}
        <BLANKLINE>
        ...
        <BLANKLINE>
                                                              r2          rmse         d2_tweedie  train_time  train_score   best_hyperparameters
        XGBRegressor (vec=None, scaler=standard, select...    0.946443    11.548574    0.904768    0:00:00     0.996710      {'colsample_bytree': 0.8229470565333281, 'gamm...
        RandomForestRegressor (vec=None, scaler=standar...    0.295580    41.882669    0.302303    0:00:00     0.357833      {'bootstrap': False, 'criterion': 'friedman_ms...
        DecisionTreeRegressor (vec=None, scaler=standar...    0.276265    42.452967    0.289315    0:00:00     0.314012      {'criterion': 'squared_error', 'max_depth': 12...
        ExtraTreesRegressor (vec=None, scaler=standard,...    0.193540    44.813584    0.187723    0:00:00     0.222682      {'bootstrap': False, 'criterion': 'friedman_ms...
        ...
        """
        return super().find_best_model_smac(x_train, y_train, x_test, y_test, n_trails=n_trails, cv_num=cv_num, scoring=scoring, small_data_eval=small_data_eval, walltime_limit_per_modeltype=walltime_limit_per_modeltype, smac_log_level=smac_log_level)
    
    def find_best_model_mass_search(self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        n_trails: int = 10,
        scoring: Literal["r2", "rmse", "d2_tweedie"] | Callable[[list[float], list[float]], float] = "r2",
        leave_loadbar: bool = True,
        save_results_path: str | None = "find_best_model_mass_search_results.csv",
    ) -> tuple[str, dict[str, float]]:
        """
        Function to run a successive halving hyperparameter search for every model

        It uses the ``warm_start`` parameter of the model and is an own implementation.
        Recommended to use as a fast method to narrow down different preprocessing steps and model combinations, but ``find_best_model_sma`` or ``randomCVsearch`` return better results.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train and optimise the models
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate the models
        n_trails : int, default=10
            max number of parameter sets to test for each model
        scoring : {"r2", "rmse", "d2_tweedie"} or callable (custom score), default="r2"
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
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
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_regression(n_samples=3000, n_features=4, noise=1, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(abs(y))
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # start modelling
        >>> from sam_ml.models.automl import RTest
        >>> 
        >>> # initialise auot-ml class
        >>> rtest = RTest(models = "all", scaler = "standard")
        >>> 
        >>> # start find_best_model_mass_search with 10 configurations per model type and find the best combination
        >>> best_model_name, score = rtest.find_best_model_mass_search(x_train,y_train,x_test,y_test, scoring="r2", n_trails=5)
        2023-12-18 11:39:12,599 - sam_ml.automl.main_auto_ml - WARNING - modeltype in 'DecisionTreeRegressor (vec=None, scaler=standard, selector=None, sampler=None)' is not supported for this search -> will be skipped
        2023-12-18 11:39:12,618 - sam_ml.automl.main_auto_ml - WARNING - modeltype in 'LassoLarsCV (vec=None, scaler=standard, selector=None, sampler=None)' is not supported for this search -> will be skipped
        ...
        2023-12-18 11:40:04,678 - sam_ml.automl.main_auto_ml - INFO - total number of models: 50
        2023-12-18 11:40:04,678 - sam_ml.automl.main_auto_ml - INFO - split number: 5, split_size (x_train): 400
        2023-12-18 11:40:04,685 - sam_ml.automl.main_auto_ml - INFO - split 1: length x_train/y_train 400/400, length x_test/y_test 2000/2000
        2023-12-18 11:40:05,032 - sam_ml.automl.main_auto_ml - INFO - new best r2: -1 -> 0.5096384300695788 (RandomForestRegressor (vec=None, scaler=standard, selector=None, sampler=None) {'bootstrap': True, 'criterion': 'squared_error', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100})
        2023-12-18 11:40:08,302 - sam_ml.automl.main_auto_ml - INFO - new best r2: 0.5096384300695788 -> 0.8496957972325866 (XGBRegressor (vec=None, scaler=standard, selector=None, sampler=None) {'colsample_bytree': 1.0, 'gamma': 0.0, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 1.0})
        ...
        2023-12-18 11:40:09,366 - sam_ml.automl.main_auto_ml - INFO - Split scores (top 5):
                                                              r2         rmse         d2_tweedie   train_time  train_score
        XGBRegressor (vec=None, scaler=standard, select...    0.849696   18.83303     0.793563     0:00:00     0.994897
        XGBRegressor (vec=None, scaler=standard, select...    0.80944    21.205614    -1.0         0:00:00     0.977253
        ...
        2023-12-18 11:40:09,368 - sam_ml.automl.main_auto_ml - INFO - removed 25 models
        2023-12-18 11:40:09,369 - sam_ml.automl.main_auto_ml - INFO - split 2: length x_train/y_train 400/400, length x_test/y_test 1600/1600
        2023-12-18 11:40:09,429 - sam_ml.automl.main_auto_ml - INFO - new best r2: -1 -> 0.5117121227247716 (RandomForestRegressor (vec=None, scaler=standard, selector=None, sampler=None) {'bootstrap': True, 'criterion': 'squared_error', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100})
        2023-12-18 11:40:10,413 - sam_ml.automl.main_auto_ml - INFO - new best r2: 0.5117121227247716 -> 0.8989539000553437 (XGBRegressor (vec=None, scaler=standard, selector=None, sampler=None) {'colsample_bytree': 1.0, 'gamma': 0.0, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 1.0})
        ...
        2023-12-18 11:40:11,134 - sam_ml.automl.main_auto_ml - INFO - removed 13 models
        2023-12-18 11:40:11,135 - sam_ml.automl.main_auto_ml - INFO - split 3: length x_train/y_train 400/400, length x_test/y_test 1200/1200
        ...
        2023-12-18 11:40:12,401 - sam_ml.automl.main_auto_ml - INFO - removed 6 models
        2023-12-18 11:40:12,402 - sam_ml.automl.main_auto_ml - INFO - split 4: length x_train/y_train 400/400, length x_test/y_test 800/800
        ...
        2023-12-18 11:40:13,327 - sam_ml.automl.main_auto_ml - INFO - removed 3 models
        2023-12-18 11:40:13,328 - sam_ml.automl.main_auto_ml - INFO - split 5: length x_train/y_train 400/400, length x_test/y_test 400/400
        ...
        2023-12-18 11:40:13,706 - sam_ml.automl.main_auto_ml - INFO - removed 2 models
        2023-12-18 11:40:13,707 - sam_ml.automl.main_auto_ml - INFO - Evaluating best model:
        <BLANKLINE>
        XGBRegressor (vec=None, scaler=standard, selector=None, sampler=None) {'colsample_bytree': 0.9330880728874675, 'gamma': 2.7381801866358395, 'learning_rate': 0.08810003129071789, 'max_depth': 10, 'min_child_weight': 10, 'n_estimators': 314, 'reg_alpha': 49, 'reg_lambda': 0.7722447692966574}
        <BLANKLINE>
        r2: 0.8767084599472158
        rmse: 17.52205436361105
        d2_tweedie: 0.8295259477160681
        """
        return super().find_best_model_mass_search(x_train, y_train, x_test, y_test, n_trails=n_trails, scoring=scoring, leave_loadbar=leave_loadbar, save_results_path=save_results_path)
