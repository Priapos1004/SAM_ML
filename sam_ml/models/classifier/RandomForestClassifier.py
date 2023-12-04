from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer, Normal
from sklearn.ensemble import RandomForestClassifier

from sam_ml.config import get_n_jobs

from ..main_classifier import Classifier


class RFC(Classifier):
    """ RandomForestClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "RandomForestClassifier",
        n_jobs: int = get_n_jobs(), 
        random_state: int = 42,
        **kwargs,
    ):
        """
        Parameters (important one)
        --------------------------
        n_estimator : int
            number of trees
        max_depth : int,
            maximum number of levels in tree
        max_features : float, int, or str,
            number of features to consider at every split
        min_samples_split : float or int,
            minimum number of samples required to split a node
        min_samples_leaf : float or int,
            minimum number of samples required at each leaf node
        bootstrap : bool,
            method of selecting samples for training each tree
        criterion : str,
            function to measure the quality of a split
        random_state : int, \
                default=42
            random_state for model
        """
        model_type = "RFC"
        model = RandomForestClassifier(
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )
        grid = ConfigurationSpace(
            seed=42,
            space={
            "n_estimators": Integer("n_estimators", (10, 1000), log=True, default=100),
            "max_depth": Integer("max_depth", (3, 15), distribution=Normal(5, 3), default=5),
            "min_samples_split": Integer("min_samples_split", (2, 10), default=2),
            "min_samples_leaf": Integer("min_samples_leaf", (1, 4), default=1),
            "bootstrap": Categorical("bootstrap", [True, False], default=True),
            "criterion": Categorical("criterion", ["gini", "entropy"], default="gini"),
            "min_weight_fraction_leaf": Float("min_weight_fraction_leaf", (0, 0.5), default=0),
            })
        
        # workaround for now -> Problems with Normal distribution (in smac_search) (04/07/2023)
        self.smac_grid = ConfigurationSpace(
            seed=42,
            space={
            "n_estimators": Integer("n_estimators", (10, 1000), log=True, default=100),
            "max_depth": Integer("max_depth", (3, 15), default=5),
            "min_samples_split": Integer("min_samples_split", (2, 10), default=2),
            "min_samples_leaf": Integer("min_samples_leaf", (1, 4), default=1),
            "bootstrap": Categorical("bootstrap", [True, False], default=True),
            "criterion": Categorical("criterion", ["gini", "entropy"], default="gini"),
            "min_weight_fraction_leaf": Float("min_weight_fraction_leaf", (0, 0.5), default=0),
            })
        super().__init__(model, model_name, model_type, grid)
