import os
import sys
import warnings
from datetime import timedelta
from inspect import isfunction
from typing import Callable, Literal

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    make_scorer,
    precision_score,
    recall_score,
)

from sam_ml.config import (
    get_avg,
    get_pos_label,
    get_scoring,
    get_secondary_scoring,
    get_strength,
    setup_logger,
)

from .main_model import Model
from .scorer import l_scoring, s_scoring

logger = setup_logger(__name__)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affects subprocesses


class Classifier(Model):
    """ Classifier parent class {abstract} """

    def __init__(self, model_object, model_name: str, model_type: str, grid: ConfigurationSpace):
        """
        Parameters
        ----------
        model_object : classifier object
            model with 'fit', 'predict', 'set_params', and 'get_params' method (see sklearn API)
        model_name : str
            name of the model
        model_type : str
            kind of estimator (e.g. 'RFC' for RandomForestClassifier)
        grid : ConfigurationSpace
            hyperparameter grid for the model
        """
        super().__init__(model_object, model_name, model_type, grid)

    def _get_score(
        self,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float],
        y_test: pd.Series,
        pred: list,
        avg: str | None,
        pos_label: int | str,
        secondary_scoring: Literal["precision", "recall"] | None,
        strength: int,
    ) -> float:
        """ 
        Calculate a score for given y true and y prediction values

        Parameters
        ----------
        scoring : {"accuracy", "precision", "recall", "s_score", "l_score"} or callable (custom score)
            metrics to evaluate the models

            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`
        y_test, pred : pd.Series, pd.Series
            Data to evaluate model
        avg : {"micro", "macro", "binary", "weighted"} or None
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None
            weights the scoring (only for "s_score"/"l_score")
        strength : int
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")

        Returns
        -------
        score : float 
            metrics score value
        """
        if scoring == "accuracy":
            score = accuracy_score(y_test, pred)
        elif scoring == "precision":
            score = precision_score(y_test, pred, average=avg, pos_label=pos_label)
        elif scoring == "recall":
            score = recall_score(y_test, pred, average=avg, pos_label=pos_label)
        elif scoring == "s_score":
            score = s_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        elif scoring == "l_score":
            score = l_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        elif isfunction(scoring):
            score = scoring(y_test, pred)
        else:
            raise ValueError(f"scoring='{scoring}' is not supported -> only  'accuracy', 'precision', 'recall', 's_score', or 'l_score'")

        return score
    
    def _get_all_scores(
        self,
        y_test: pd.Series,
        pred: list,
        avg: str | None,
        pos_label: int | str,
        secondary_scoring: Literal["precision", "recall"] | None,
        strength: int,
        custom_score: Callable[[list[int], list[int]], float] | None,
    ) -> dict[str, float]:
        """ 
        Calculate accuracy, precision, recall, s_score, l_score, and optional custom_score metrics

        Parameters
        ----------
        y_test, pred : pd.Series, pd.Series
            Data to evaluate model
        avg : {"micro", "macro", "binary", "weighted"} or None
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None
            weights the scoring (only for "s_score"/"l_score")
        strength : int
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...}

            or if ``custom_score != None``:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'custom_score': ...,}
        """
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average=avg, pos_label=pos_label)
        recall = recall_score(y_test, pred, average=avg, pos_label=pos_label)
        s_score = s_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)
        l_score = l_scoring(y_test, pred, strength=strength, scoring=secondary_scoring, pos_label=pos_label)

        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "s_score": s_score,
            "l_score": l_score,
        }

        if isfunction(custom_score):
            custom_scores = custom_score(y_test, pred)
            scores["custom_score"] = custom_scores

        return scores
    
    def _print_scores(self, scores: dict, y_test: pd.Series, pred: list):
        """
        Function to print out the values of a dictionary

        Parameters
        ----------
        scores: dict
            dictionary with score names and values
        y_test : pd.Series
            true y values
        pred : list
            predicted y values of model

        Returns
        -------
        key-value pairs in console, format: 
        
        key1: value1

        key2: value2
        
        ...
        """
        for key in scores:
            print(f"{key}: {scores[key]}")
        print()
        print("classification report:")
        print(classification_report(y_test, pred))

    def _make_scorer(
        self,
        avg: str | None,
        pos_label: int | str,
        secondary_scoring: Literal["precision", "recall"] | None,
        strength: int,
        custom_score: Callable[[list[int], list[int]], float] | None,
    ) -> dict[str, Callable]:
        """
        Function to create a dictionary with scorer for the crossvalidation
        
        Parameters
        ----------
        avg : {"micro", "macro", "binary", "weighted"} or None
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None
            weights the scoring (only for "s_score"/"l_score")
        strength : int
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scorer : dict[str, Callable]
            dictionary with scorer functions
        """
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

        if isfunction(custom_score):
            scorer["custom_score"] = make_scorer(custom_score)

        return scorer
    
    def _make_cv_scores(
            self,
            score: dict,
            custom_score: Callable[[list[int], list[int]], float] | None,
    ) -> dict[str, float]:
        """
        Function to create from the crossvalidation results a dictionary
        
        Parameters
        ----------
        score : dict
            crossvalidation average column results
        custom_score : callable or None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        cv_scores : dict
            restructured dictionary
        """
        cv_scores = {
            "accuracy": score[list(score.keys())[6]],
            "precision": score[list(score.keys())[2]],
            "recall": score[list(score.keys())[4]],
            "s_score": score[list(score.keys())[8]],
            "l_score": score[list(score.keys())[10]],
            "train_time": str(timedelta(seconds = round(score[list(score.keys())[0]]))),
            "train_score": score[list(score.keys())[7]],
        }

        if isfunction(custom_score):
            cv_scores["custom_score"] = score[list(score.keys())[12]]
        
        return cv_scores
    
    @staticmethod
    def _create_prediction_proba(pred_proba: np.ndarray, probability: float) -> tuple[list[int], dict]:
        """
        Function to convert probability of classes into class numbers

        Parameters
        ----------
        pred_proba : np.ndarray
            np.ndarray with probability for every class per datapoint
        probability: float (0 to 1)
            probability for class 1 (with value 0.5 is like ``evaluate_score`` method). With increasing the probability parameter, precision will likely increase and recall will decrease (with decreasing the probability parameter, the otherway around).
        
        Returns
        -------
        pred : list[int]
            list with class numbers
        proba_stats : dict
            dictionary with stats of probabilities for class 1, format:

            {'min proba': ...,
            'max proba': ...,
            'mean proba': ...,
            'median proba': ...,
            'std proba': ...}

        """
        class_1_proba = pred_proba[:, 1]
        pred = [int(x > probability) for x in class_1_proba]
        proba_stats = {
            "min proba": np.min(class_1_proba),
            "max proba": np.max(class_1_proba),
            "mean proba": np.mean(class_1_proba),
            "median proba": np.median(class_1_proba),
            "std proba": np.std(class_1_proba),
        }
        return pred, proba_stats

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series, 
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str | None = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        console_out: bool = True
    ) -> tuple[float, str]:
        """
        Function to train the model

        Every classifier has a train- and fit-method. They both use the fit-method of the wrapped model, 
        but the train-method returns the train time and the train score of the model. 

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train model
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
        console_out : bool, default=True
            shall the score and time be printed out

        Returns
        -------
        train_score : float 
            train score value
        train_time : str
            train time in format: "0:00:00" (hours:minutes:seconds)

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>>
        >>> # train model
        >>> from sam_ml.models.classifier import LR
        >>> 
        >>> model = LR()
        >>> model.train(X, y)
        Train score: 0.9891840171120917 - Train time: 0:00:02
        """
        return super().train(x_train, y_train, console_out, scoring=scoring, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
    
    def train_warm_start(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series, 
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str | None = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        console_out: bool = True
    ) -> tuple[float, str]:
        """
        Function to warm_start train the model

        This function only differs for pipeline objects (with preprocessing) from the train method.
        For pipeline objects, it only traines the preprocessing steps the first time and then only uses them to preprocess.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to train model
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
        console_out : bool, default=True
            shall the score and time be printed out

        Returns
        -------
        train_score : float 
            train score value
        train_time : str
            train time in format: "0:00:00" (hours:minutes:seconds)

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>>
        >>> # train model
        >>> from sam_ml.models.classifier import LR
        >>> 
        >>> model = LR()
        >>> model.train_warm_start(X, y)
        Train score: 0.9891840171120917 - Train time: 0:00:02
        """
        return super().train_warm_start(x_train, y_train, console_out, scoring=scoring, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)

    def evaluate(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        console_out: bool = True,
        avg: str | None = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        custom_score: Callable[[list[int], list[int]], float] | None = None,
    ) -> dict[str, float]:
        """
        Function to create multiple scores with predict function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate model
        console_out : bool, default=True
            shall the result of the different scores and a classification_report be printed into the console
        avg : {"micro", "macro", "binary", "weighted"} or None, default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        custom_score : callable or None, default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...}

            or if ``custom_score != None``:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'custom_score': ...,}

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # train and evaluate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> model.train(x_train, y_train)
        >>> scores = model.evaluate(x_test, y_test)
        Train score: 0.9891840171120917 - Train time: 0:00:02
        accuracy: 0.802
        precision: 0.8030604133545309
        recall: 0.7957575757575757
        s_score: 0.9395778023942218
        l_score: 0.9990945415060262
        <BLANKLINE>
        classification report: 
                        precision   recall  f1-score    support
        <BLANKLINE>
                0       0.81        0.73    0.77        225
                1       0.80        0.86    0.83        275
        <BLANKLINE>
        accuracy                            0.80        500
        macro avg       0.80        0.80    0.80        500
        weighted avg    0.80        0.80    0.80        500
        <BLANKLINE>
        """
        return super().evaluate(x_test, y_test, console_out=console_out, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength, custom_score=custom_score)
    
    def evaluate_score(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str | None = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
    ) -> float:
        """
        Function to create a score with predict function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate model
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
        score : float 
            metrics score value

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # train and evaluate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> model.fit(x_train, y_train)
        >>> recall = model.evaluate_score(x_test, y_test, scoring="recall")
        >>> print(f"recall: {recall}")
        recall: 0.4
        """
        return super().evaluate_score(scoring, x_test, y_test, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
    
    def evaluate_proba(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        console_out: bool = True,
        avg: str | None = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        custom_score: Callable[[list[int], list[int]], float] | None = None,
        probability: float = 0.5,
    ) -> dict[str, float]:
        """
        Function to create multiple scores for binary classification with predict_proba function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate model
        console_out : bool, default=True
            shall the result of the different scores and a classification_report be printed. Also prints stats for the probabilities
        avg : {"micro", "macro", "binary", "weighted"} or None, default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        custom_score : callable or None, default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.
        probability: float (0 to 1), default=0.5
            probability for class 1 (with value 0.5 is like ``evaluate_score`` method). With increasing the probability parameter, precision will likely increase and recall will decrease (with decreasing the probability parameter, the otherway around).

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...}

            or if ``custom_score != None``:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'custom_score': ...,}

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=3000, n_features=4, n_classes=2, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(y)
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # train and evaluate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> model.train(x_train, y_train)
        >>> scores = model.evaluate_proba(x_test, y_test, probability=0.4)
        Train score: 0.9775 - Train time: 0:00:00
        accuracy: 0.9733333333333334
        precision: 0.9728695961572674
        recall: 0.9742405994915028
        s_score: 0.9930964441542017
        l_score: 0.9999999991441061
        min proba: 5.126066053780961e-12
        max proba: 0.9999731025066587
        mean proba: 0.4701783612343521
        median proba: 0.11068735707926472
        std proba: 0.474678546763958
        <BLANKLINE>
        classification report:
                        precision   recall  f1-score   support
        <BLANKLINE>
                0       0.99        0.96    0.97       318
                1       0.96        0.99    0.97       282
        <BLANKLINE>
        accuracy                            0.97       600
        macro avg       0.97        0.97    0.97       600
        weighted avg    0.97        0.97    0.97       600
        <BLANKLINE>
        """
        if len(set(y_test)) != 2:
            raise ValueError(f"Expected binary classification data, but received y_test with {len(set(y_test))} classes")

        pred_proba = self.predict_proba(x_test)
        pred, proba_stats = self._create_prediction_proba(pred_proba, probability=probability)
        scores = self._get_all_scores(y_test, pred, avg, pos_label, secondary_scoring, strength, custom_score)

        if console_out:
            self._print_scores(scores | proba_stats, y_test, pred)

        return scores
    
    def evaluate_score_proba(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str | None = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        probability: float = 0.5,
    ) -> float:
        """
        Function to create a score for binary classification with predict_proba function of model

        Parameters
        ----------
        x_test, y_test : pd.DataFrame, pd.Series
            Data to evaluate model
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
        probability: float (0 to 1), default=0.5
            probability for class 1 (with value 0.5 is like ``evaluate_score`` method). With increasing the probability parameter, precision will likely increase and recall will decrease (with decreasing the probability parameter, the otherway around).

        Returns
        -------
        score : float 
            metrics score value

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=3000, n_features=4, n_classes=2, random_state=42)
        >>> X, y = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"]), pd.Series(y)
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>> 
        >>> # train and evaluate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> model.fit(x_train, y_train)
        >>> recall = model.evaluate_score_proba(x_test, y_test, scoring="recall", probability=0.4)
        >>> print(f"recall: {recall}")
        recall: 0.9742405994915028
        """
        if len(set(y_test)) != 2:
            raise ValueError(f"Expected binary classification data, but received y_test with {len(set(y_test))} classes")

        pred_proba = self.predict_proba(x_test)
        pred, _ = self._create_prediction_proba(pred_proba, probability=probability)
        score = self._get_score(scoring, y_test, pred, avg, pos_label, secondary_scoring, strength)

        return score

    def cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_num: int = 10,
        console_out: bool = True,
        avg: str | None = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        custom_score: Callable[[list[int], list[int]], float] | None = None,
    ) -> dict[str, float]:
        """
        Random split crossvalidation

        Parameters
        ----------
        X, y : pd.DataFrame, pd.Series
            Data to cross validate on
        cv_num : int, default=10
            number of different random splits
        console_out : bool, default=True
            shall the result dataframe of the different scores for the different runs be printed
        avg : {"micro", "macro", "binary", "weighted"} or None, default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        custom_score : callable or None, default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'train_time': ...,
                'train_score': ...,}

            or if ``custom_score != None``:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'train_time': ...,
                'train_score': ...,
                'custom_score': ...,}

        The scores are also saved in ``self.cv_scores``.

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>> 
        >>> # cross validate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> scores = model.cross_validation(X, y, cv_num=3)
        <BLANKLINE>
                                    0         1         2           average
        fit_time                    1.194662  1.295036  1.210156    1.233285
        score_time                  0.167266  0.149569  0.173546    0.163460
        test_precision (macro)      0.779381  0.809037  0.761263    0.783227
        train_precision (macro)     0.951738  0.947397  0.943044    0.947393
        test_recall (macro)         0.774488  0.800144  0.761423    0.778685
        train_recall (macro)        0.948928  0.943901  0.940066    0.944298
        test_accuracy               0.776978  0.803121  0.762305    0.780802
        train_accuracy              0.950180  0.945411  0.941212    0.945601
        test_s_score                0.923052  0.937806  0.917214    0.926024
        train_s_score               0.990794  0.990162  0.989660    0.990206
        test_l_score                0.998393  0.998836  0.998575    0.998602
        train_l_score               1.000000  1.000000  1.000000    1.000000
        """
        return super().cross_validation(X, y, cv_num=cv_num, console_out=console_out, custom_score=custom_score, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)

    def cross_validation_small_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        leave_loadbar: bool = True,
        console_out: bool = True,
        avg: str | None = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        custom_score: Callable[[list[int], list[int]], float] | None = None,
    ) -> dict[str, float]:
        """
        One-vs-all cross validation for small datasets

        In the cross_validation_small_data-method, the model will be trained on all datapoints except one and then tested on this last one. 
        This will be repeated for all datapoints so that we have our predictions for all datapoints.

        Advantage: optimal use of information for training

        Disadvantage: long train time

        This concept is very useful for small datasets (recommended: datapoints < 150) because the long train time is still not too long and 
        especially with a small amount of information for the model, it is important to use all the information one has for the training.

        Parameters
        ----------
        X, y : pd.DataFrame, pd.Series
            Data to cross validate on
        leave_loadbar : bool, default=True
            shall the loading bar of the training be visible after training (True - load bar will still be visible)
        console_out : bool, default=True
            shall the result of the different scores and a classification_report be printed into the console
        avg : {"micro", "macro", "binary", "weighted"} or None, default="macro"
            average to use for precision and recall score. If ``None``, the scores for each class are returned.
        pos_label : int or str, default=-1
            if ``avg="binary"``, pos_label says which class to score. pos_label is used by s_score/l_score
        secondary_scoring : {"precision", "recall"} or None, default=None
            weights the scoring (only for "s_score"/"l_score")
        strength : int, default=3
            higher strength means a higher weight for the preferred secondary_scoring/pos_label (only for "s_score"/"l_score")
        custom_score : callable or None, default=None
            custom score function (or loss function) with signature
            `score_func(y, y_pred, **kwargs)`

            If ``None``, no custom score will be calculated and also the key "custom_score" does not exist in the returned dictionary.

        Returns
        -------
        scores : dict 
            dictionary of format:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'train_time': ...,
                'train_score': ...,}

            or if ``custom_score != None``:

                {'accuracy': ...,
                'precision': ...,
                'recall': ...,
                's_score': ...,
                'l_score': ...,
                'train_time': ...,
                'train_score': ...,
                'custom_score': ...,}

        The scores are also saved in ``self.cv_scores``.

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>> 
        >>> # cross validate model
        >>> from sam_ml.models.classifier import LR
        >>>
        >>> model = LR()
        >>> scores = model.cross_validation_small_data(X, y)
        accuracy: 0.7
        precision: 0.7747221430607011
        recall: 0.672883787661406
        s_score: 0.40853182756324635
        l_score: 0.7812935895658734
        train_time: 0:00:00
        train_score: 0.9946286670687757
        <BLANKLINE>
        classification report:
                        precision   recall  f1-score    support
        <BLANKLINE>
                0       0.65        0.96    0.78        82
                1       0.90        0.38    0.54        68
        <BLANKLINE>
        accuracy                            0.70        150
        macro avg       0.77        0.67    0.66        150
        weighted avg    0.76        0.70    0.67        150
        <BLANKLINE>
        """
        return super().cross_validation_small_data(X,y,leave_loadbar=leave_loadbar, console_out=console_out, custom_score=custom_score, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
    
    def smac_search(
        self,
        x_train: pd.DataFrame, 
        y_train: pd.Series,
        n_trails: int = 50,
        cv_num: int = 5,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str | None = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        small_data_eval: bool = False,
        walltime_limit: int = 600,
        log_level: int = 20,
    ) -> Configuration:
        """
        Hyperparametertuning with SMAC library HyperparameterOptimizationFacade [can only be used in the sam_ml version with swig]

        The smac_search-method will more "intelligent" search your hyperparameter space than the randomCVsearch and 
        returns the best hyperparameter set. Additionally to the n_trails parameter, it also takes a walltime_limit parameter 
        that defines the maximum time in seconds that the search will take.

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to cross validate on
        n_trails : int, default=50
            max number of parameter sets to test
        cv_num : int, default=5
            number of different random splits
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
        walltime_limit : int, default=600
            the maximum time in seconds that SMAC is allowed to run
        log_level : int, default=20
            10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR, 50 - CRITICAL (SMAC3 library log levels)

        Returns
        -------
        incumbent : ConfigSpace.Configuration
            ConfigSpace.Configuration with best hyperparameters (can be used like dict)

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>>
        >>> # use smac_search
        >>> from sam_ml.models.classifier import LR
        >>> 
        >>> model = LR()
        >>> best_hyperparam = model.smac_search(X, y, n_trails=20, cv_num=5, scoring="recall")
        >>> print(f"best hyperparameters: {best_hyperparam}")
        [INFO][abstract_initial_design.py:82] Using `n_configs` and ignoring `n_configs_per_hyperparameter`.
        [INFO][abstract_initial_design.py:147] Using 2 initial design configurations and 0 additional configurations.
        [INFO][abstract_initial_design.py:147] Using 3 initial design configurations and 0 additional configurations.
        [INFO][abstract_intensifier.py:305] Using only one seed for deterministic scenario.
        [INFO][abstract_intensifier.py:515] Added config 12be8a as new incumbent because there are no incumbents yet.
        [INFO][abstract_intensifier.py:590] Added config ce10f4 and rejected config 12be8a as incumbent because it is not better than the incumbents on 1 instances:
        [INFO][abstract_intensifier.py:590] Added config b35335 and rejected config ce10f4 as incumbent because it is not better than the incumbents on 1 instances:
        [INFO][smbo.py:327] Configuration budget is exhausted:
        [INFO][smbo.py:328] --- Remaining wallclock time: 590.5625982284546
        [INFO][smbo.py:329] --- Remaining cpu time: inf
        [INFO][smbo.py:330] --- Remaining trials: 0
        best hyperparameters: Configuration(values={
        'C': 66.7049177605834,
        'penalty': 'l2',
        'solver': 'lbfgs',
        })
        """
        return super().smac_search(x_train, y_train, scoring=scoring, n_trails=n_trails, cv_num=cv_num, small_data_eval=small_data_eval, walltime_limit=walltime_limit, log_level=log_level, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)

    def randomCVsearch(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_trails: int = 10,
        cv_num: int = 5,
        scoring: Literal["accuracy", "precision", "recall", "s_score", "l_score"] | Callable[[list[int], list[int]], float] = get_scoring(),
        avg: str | None = get_avg(),
        pos_label: int | str = get_pos_label(),
        secondary_scoring: Literal["precision", "recall"] | None = get_secondary_scoring(),
        strength: int = get_strength(),
        small_data_eval: bool = False,
        leave_loadbar: bool = True,
    ) -> tuple[dict, float]:
        """
        Hyperparametertuning with randomCVsearch

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            Data to cross validate on
        n_trails : int, default=10
            max number of parameter sets to test
        cv_num : int, default=5
            number of different random splits
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
            shall the loading bar of the different parameter sets be visible after training (True - load bar will still be visible)

        Returns
        -------
        best_hyperparameters : dict
            best hyperparameter set
        best_score : float
            the score of the best hyperparameter set

        Notes
        -----
        if you interrupt the keyboard during the run of randomCVsearch, the interim result will be returned

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>>
        >>> # initialise model
        >>> from sam_ml.models.classifier import LR
        >>> model = LR()
        >>>
        >>> # use randomCVsearch
        >>> best_hyperparam, best_score = model.randomCVsearch(X, y, n_trails=20, cv_num=5, scoring="recall")
        >>> print(f"best hyperparameters: {best_hyperparam}, best score: {best_score}")
        best hyperparameters: {'C': 8.471801418819979, 'penalty': 'l2', 'solver': 'newton-cg'}, best score: 0.765
        """
        return super().randomCVsearch(x_train, y_train, n_trails=n_trails, cv_num=cv_num, scoring=scoring, small_data_eval=small_data_eval, leave_loadbar=leave_loadbar, avg=avg, pos_label=pos_label, secondary_scoring=secondary_scoring, strength=strength)
