from ConfigSpace import Categorical, ConfigurationSpace, Integer, Normal
from sklearn.ensemble import ExtraTreesClassifier

from .main_classifier import Classifier


class ETC(Classifier):
    """ ExtraTreesClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "ExtraTreesClassifier",
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            n_estimators: Number of trees
            max_depth: Maximum number of levels in tree
            n_jobs: how many cores shall be used (-1 means all)
            random_state: random_state for model
            verbose: log level (higher number --> more logs)
            warm_start: work with previous fit and add more estimator

            max_features: Number of features to consider at every split
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at each leaf node
            bootstrap: Method of selecting samples for training each tree
            criterion: function to measure the quality of a split
        """
        model_type = "ETC"
        model = ExtraTreesClassifier(
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )
        grid = ConfigurationSpace(
            seed=42,
            space={
            "n_estimators": Integer("n_estimators", (1, 1000), log=True),
            "max_depth": Integer("max_depth", (2, 15), distribution=Normal(5, 3)),
            "min_samples_split": Integer("min_samples_split", (2, 10)),
            "min_samples_leaf": Integer("min_samples_leaf", (1, 4)),
            "bootstrap": Categorical("bootstrap", [True, False], default=False),
            "criterion": Categorical("criterion", ["gini", "entropy"]),
            })
        super().__init__(model, model_name, model_type, grid)
