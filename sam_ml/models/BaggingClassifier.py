import warnings

from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .main_classifier import Classifier

warnings. filterwarnings('ignore')


class BC(Classifier):
    """ BaggingClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "BaggingClassifier",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        @param (important one):
            base_estimator: base estimator from which the boosted ensemble is built (default: DecisionTreeClassifier with max_depth=1)
            n_estimator: number of boosting stages to perform
            max_samples: the number of samples to draw from X to train each base estimator
            max_features: the number of features to draw from X to train each base estimator
            bootstrap: whether samples are drawn with replacement. If False, sampling without replacement is performed
            bootstrap_features: whether features are drawn with replacement
        """
        model_type = "BC"
        model = BaggingClassifier(
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        if type(model.base_estimator) == RandomForestClassifier:
            grid = {
                "base_estimator": [RandomForestClassifier(max_depth=i, n_estimators=j) for i in range(1,11) for j in (5, 10, 20, 50, 100)]+[SVC(probability=True, kernel='linear'), LogisticRegression(), GradientBoostingClassifier(), KNeighborsClassifier()],
                "n_estimators": list(range(10, 101, 10)) + [3, 4, 5, 6, 7, 8, 9, 200, 500, 1000, 1500, 3000],
                "max_samples": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "max_features": [0.5, 1.0, 2, 4],
                "bootstrap": [True, False],
                "bootstrap_features": [True, False],
            }
        else:
            grid = {
                "base_estimator": [DecisionTreeClassifier(max_depth=i) for i in range(1,11)]+[SVC(probability=True, kernel='linear'), LogisticRegression(), GradientBoostingClassifier(), KNeighborsClassifier()],
                "n_estimators": list(range(10, 101, 10)) + [3, 4, 5, 6, 7, 8, 9, 200, 500, 1000, 1500, 3000],
                "max_samples": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "max_features": [0.5, 1.0, 2, 4],
                "bootstrap": [True, False],
                "bootstrap_features": [True, False],
            }
        super().__init__(model, model_name, model_type, grid)
