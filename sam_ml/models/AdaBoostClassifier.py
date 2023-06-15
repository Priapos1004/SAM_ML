from numpy import arange
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
            base_estimator: base estimator from which the boosted ensemble is built (default: DecisionTreeClassifier with max_depth=1)
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
        if type(model.base_estimator) == RandomForestClassifier:
            grid = {
                "base_estimator": [RandomForestClassifier(max_depth=i, n_estimators=j) for i in range(1,11) for j in (5, 10, 20, 50, 100)]+[SVC(probability=True, kernel='linear'), LogisticRegression(), GradientBoostingClassifier()],
                "n_estimators": list(range(10, 101, 10)) + [200, 500, 1000, 1500, 3000],
                "learning_rate": [0.1, 0.05, 0.01, 0.005]+list(arange(0.2,2.1,0.1)),
                "algorithm": ["SAMME.R", "SAMME"],
            }
        else:
            grid = {
                "base_estimator": [DecisionTreeClassifier(max_depth=i) for i in range(1,11)]+[SVC(probability=True, kernel='linear'), LogisticRegression(), GradientBoostingClassifier()],
                "n_estimators": list(range(10, 101, 10)) + [200, 500, 1000, 1500, 3000],
                "learning_rate": [0.1, 0.05, 0.01, 0.005]+list(arange(0.2,2.1,0.1)),
                "algorithm": ["SAMME.R", "SAMME"],
            }
        super().__init__(model, model_name, model_type, grid)
