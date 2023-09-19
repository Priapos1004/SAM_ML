from ConfigSpace import Beta, Categorical, ConfigurationSpace, Float, Integer
from sklearn.base import ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ..main_classifier import Classifier


class ABC(Classifier):
    """ AdaBoostClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "AdaBoostClassifier",
        random_state: int = 42,
        estimator: str | ClassifierMixin = "DTC",
        **kwargs,
    ):
        """
        @param (important one):
            estimator: base estimator from which the boosted ensemble is built (default: DecisionTreeClassifier with max_depth=1), also possible is string 'DTC', 'RFC', and 'LR'
            n_estimator: number of boosting stages to perform
            learning_rate: shrinks the contribution of each tree by learning rate
            algorithm: boosting algorithm
            random_state: random_state for model
        """
        model_type = "ABC"

        kwargs_estimator = {}
        kwargs_ABC = {}
        for key in kwargs:
            if key.startswith("estimator__"):
                new_key = key.removeprefix("estimator__")
                kwargs_estimator[new_key] = kwargs[key]
            else:
                kwargs_ABC[key] = kwargs[key]

        if type(estimator) == str:
            model_name += f" ({estimator} based)"
            if estimator == "DTC":
                if not kwargs_estimator:
                    estimator = DecisionTreeClassifier(max_depth=1)
                else:
                    estimator = DecisionTreeClassifier(**kwargs_estimator)
            elif estimator == "RFC":
                if not kwargs_estimator:
                    estimator = RandomForestClassifier(max_depth=5, n_estimators=50, random_state=42)
                else:
                    if not "random_state" in kwargs_estimator:
                        estimator = RandomForestClassifier(**kwargs_estimator, random_state=42)
                    else:
                        estimator = RandomForestClassifier(**kwargs_estimator)
            elif estimator == "LR":
                if not kwargs_estimator:
                    estimator = LogisticRegression(random_state=42)
                else:
                    if not "random_state" in kwargs_estimator:
                        estimator = LogisticRegression(**kwargs_estimator, random_state=42)
                    else:
                        estimator = LogisticRegression(**kwargs_estimator)
            else:
                raise ValueError(f"invalid string input ('{estimator}') for estimator -> use 'DTC', 'RFC', or 'LR'")

        model = AdaBoostClassifier(
            random_state=random_state,
            estimator=estimator,
            **kwargs_ABC,
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
