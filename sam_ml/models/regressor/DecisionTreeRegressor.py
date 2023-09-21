from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

from ..main_regressor import Regressor


class DTR(Regressor):
    """ DecisionTreeRegressor Wrapper class """

    def __init__(
        self,
        model_name: str = "DecisionTreeRegressor",
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
        model_type = "DTR"
        model = DecisionTreeRegressor(
            random_state=random_state,
            **kwargs,
        )
        grid = ConfigurationSpace(
            seed=42,
            space={
            "splitter": Categorical("splitter", ["best", "random"], default="best"),
            "criterion": Categorical("criterion", ["friedman_mse", "squared_error"], default="squared_error"),
            "max_depth": Integer("max_depth", (1, 12), default=5),
            "min_samples_split": Integer("min_samples_split", (2, 10), default=2),
            "min_samples_leaf": Integer("min_samples_leaf", (1, 5), default=1),
            "min_weight_fraction_leaf": Float("min_weight_fraction_leaf", (0, 0.5), default=0),
            "max_features": Categorical("max_features", ["auto","log2","sqrt"], default="auto"),
            "max_leaf_nodes": Integer("max_leaf_nodes", (10, 90), default=90),
            })
        super().__init__(model, model_name, model_type, grid)
    
    def plot_tree(self):
        return tree.plot_tree(self.model)
