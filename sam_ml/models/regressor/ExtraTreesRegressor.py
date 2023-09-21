from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer, Normal
from sklearn.ensemble import ExtraTreesRegressor

from sam_ml.config import get_n_jobs

from ..main_regressor import Regressor


class ETR(Regressor):
    """ ExtraTreesRegressor Wrapper class """

    def __init__(
        self,
        model_name: str = "ExtraTreesRegressor",
        n_jobs: int = get_n_jobs(),
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
        model_type = "ETR"
        model = ExtraTreesRegressor(
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
            "bootstrap": Categorical("bootstrap", [True, False], default=False),
            "criterion": Categorical("criterion", ["friedman_mse", "squared_error"], default="squared_error"),
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
            "bootstrap": Categorical("bootstrap", [True, False], default=False),
            "criterion": Categorical("criterion", ["friedman_mse", "squared_error"], default="squared_error"),
            "min_weight_fraction_leaf": Float("min_weight_fraction_leaf", (0, 0.5), default=0),
            })
        super().__init__(model, model_name, model_type, grid)
