from ConfigSpace import Categorical, ConfigurationSpace, Integer
from sklearn.tree import DecisionTreeClassifier

from .main_classifier import Classifier


class DTC(Classifier):
    """ DecisionTreeClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "DecisionTreeClassifier",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            criterion: function to measure the quality of a split
            max_depth: Maximum number of levels in tree
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at each leaf node
            random_state: random_state for model
        """
        model_type = "DTC"
        model = DecisionTreeClassifier(
            random_state=random_state,
            **kwargs,
        )
        grid = ConfigurationSpace(
            seed=42,
            space={
            "criterion": Categorical("criterion", ["gini", "entropy"], default="gini"),
            "max_depth": Integer("max_depth", (1, 10), default=5),
            "min_samples_split": Integer("min_samples_split", (2, 10), default=2),
            "min_samples_leaf": Integer("min_samples_leaf", (1, 5), default=1),
            })
        super().__init__(model, model_name, model_type, grid)
