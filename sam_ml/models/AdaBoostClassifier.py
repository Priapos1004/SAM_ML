from ConfigSpace import Beta, Categorical, ConfigurationSpace, Float, Integer
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .main_classifier import Classifier


class ABC(Classifier):
    """ AdaBoostClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "AdaBoostClassifier",
        random_state: int = 42,
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
            **kwargs,
        )
        if type(model.estimator) == RandomForestClassifier:
            core_estimator = [RandomForestClassifier(max_depth=i, n_estimators=j) for j in  (100, 50, 20, 10, 5) for i in range(1,11)]
        elif type(model.estimator) == DecisionTreeClassifier or model.estimator is None:
            core_estimator = [DecisionTreeClassifier(max_depth=i) for i in range(1,11)]
        else:
            core_estimator = [SVC(probability=True, kernel='linear'), GradientBoostingClassifier(), LogisticRegression()]
        
        grid = ConfigurationSpace(
            seed=42,
            space={
            "estimator": Categorical("estimator", core_estimator, default=core_estimator[2]),
            "n_estimators": Integer("n_estimators", (10, 3000), log=True, default=50),
            "learning_rate": Float("learning_rate", (0.005, 2), distribution=Beta(10, 5), default=1),
            "algorithm": Categorical("algorithm", ["SAMME.R", "SAMME"], default="SAMME.R"),
            })
        super().__init__(model, model_name, model_type, grid)
