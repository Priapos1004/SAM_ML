import warnings

from ConfigSpace import Beta, Categorical, ConfigurationSpace, Float, Integer
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
            estimator: base estimator from which the boosted ensemble is built (default: DecisionTreeClassifier with max_depth=1)
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
        if type(model.estimator) == RandomForestClassifier:
            core_estimator = [RandomForestClassifier(max_depth=i, n_estimators=j) for j in (100, 50, 20, 10, 5) for i in range(1,11)]
        elif type(model.estimator) == DecisionTreeClassifier or model.estimator is None:
            core_estimator = [DecisionTreeClassifier(max_depth=i) for i in range(1,11)]
        else:
            core_estimator = [SVC(probability=True, kernel='linear'), GradientBoostingClassifier(), KNeighborsClassifier(), LogisticRegression()]

        grid = ConfigurationSpace(
            seed=42,
            space={
            "estimator": Categorical("estimator", core_estimator, default=core_estimator[3]),
            "n_estimators": Integer("n_estimators", (3, 3000), distribution=Beta(1, 15), default=10),
            "max_samples": Float("max_samples", (0.1, 1), default=1),
            "max_features": Categorical("max_features", [0.5, 0.9, 1.0, 2, 4], default=1.0),
            "bootstrap": Categorical("bootstrap", [True, False], default=True),
            "bootstrap_features": Categorical("bootstrap_features", [True, False], default=False),
            })
        super().__init__(model, model_name, model_type, grid)
