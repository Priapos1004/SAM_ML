import warnings

from ConfigSpace import Beta, Categorical, ConfigurationSpace, Float, Integer
from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sam_ml.config import get_n_jobs

from .main_classifier import Classifier

warnings. filterwarnings('ignore')


class BC(Classifier):
    """ BaggingClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "BaggingClassifier",
        random_state: int = 42,
        n_jobs: int = get_n_jobs(),
        estimator: ClassifierMixin = DecisionTreeClassifier(max_depth=1),
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
            estimator=estimator,
            **kwargs,
        )

        grid = ConfigurationSpace(
            seed=42,
            space={
            "n_estimators": Integer("n_estimators", (3, 3000), distribution=Beta(1, 15), default=10),
            "max_samples": Float("max_samples", (0.1, 1), default=1),
            "max_features": Categorical("max_features", [0.5, 0.9, 1.0, 2, 4], default=1.0),
            "bootstrap": Categorical("bootstrap", [True, False], default=True),
            "bootstrap_features": Categorical("bootstrap_features", [True, False], default=False),
            })
        
        if type(model.estimator) == RandomForestClassifier:
            grid.add_hyperparameter(Integer("estimator__max_depth", (1, 11), default=5))
            grid.add_hyperparameter(Integer("estimator__n_estimators", (5, 100), log=True, default=50))
        elif type(model.estimator) == DecisionTreeClassifier:
            grid.add_hyperparameter(Integer("estimator__max_depth", (1, 11), default=1))
        
        super().__init__(model, model_name, model_type, grid)
