from sklearn.tree import DecisionTreeClassifier

from .main_classifier import Classifier


class DTC(Classifier):
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
        self.model_name = model_name
        self.model_type = "DTC"
        self.model = DecisionTreeClassifier(
            random_state=random_state,
            **kwargs,
        )
        self._grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": list(range(1, 10)),
            "min_samples_split": list(range(2, 10)),
            "min_samples_leaf": list(range(1, 5)),
        }
