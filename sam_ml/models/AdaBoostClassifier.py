from ConfigSpace import Beta, Categorical, ConfigurationSpace, Float, Integer
from sklearn.base import ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .main_classifier import Classifier


class ABC(Classifier):
    """ AdaBoostClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "AdaBoostClassifier",
        random_state: int = 42,
        estimator: ClassifierMixin = DecisionTreeClassifier(max_depth=1),
        **kwargs,
    ):
        """
        @param (important one):
            estimator: base estimator from which the boosted ensemble is built (default: DecisionTreeClassifier with max_depth=1)
            n_estimator: number of boosting stages to perform
            learning_rate: shrinks the contribution of each tree by learning rate
            algorithm: boosting algorithm
            random_state: random_state for model
        """
        model_type = "ABC"
        model = AdaBoostClassifier(
            random_state=random_state,
            estimator=estimator,
            **kwargs,
        )
        
        grid = ConfigurationSpace(
            seed=42,
            space={
            "n_estimators": Integer("n_estimators", (10, 3000), log=True, default=50),
            "learning_rate": Float("learning_rate", (0.005, 2), distribution=Beta(10, 5), default=1),
            "algorithm": Categorical("algorithm", ["SAMME.R", "SAMME"], default="SAMME.R"),
            })
        
        if type(model.estimator) == RandomForestClassifier:
            grid.add_hyperparameter(Integer("estimator__max_depth", (1, 11), default=5))
            grid.add_hyperparameter(Integer("estimator__n_estimators", (5, 100), log=True, default=50))
        elif type(model.estimator) == DecisionTreeClassifier:
            grid.add_hyperparameter(Integer("estimator__max_depth", (1, 11), default=1))
        
        super().__init__(model, model_name, model_type, grid)
