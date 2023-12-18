from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer, Normal
from sklearn.ensemble import GradientBoostingClassifier

from ..main_classifier import Classifier


class GBM(Classifier):
    """ GradientBoostingMachine Wrapper class """

    def __init__(
        self,
        model_name: str = "GradientBoostingMachine",
        random_state: int = 42,
        **kwargs,
    ):
        """
        Parameters (important one)
        --------------------------
        n_estimator : int
            number of boosting stages to perform
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
        subsample : float,
            fraction of samples to be used for fitting the individual base learners
        loss : str,
            the loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss 'exponential' gradient boosting recovers the AdaBoost algorithm
        learning_rate : float,
            shrinks the contribution of each tree by learning rate
        random_state : int, \
                default=42
            random_state for model
        
        Notes
        -----
        You can use all parameters of the wrapped model when initialising the wrapper class.
        """
        model_type = "GBM"
        model = GradientBoostingClassifier(random_state=random_state, **kwargs,)
        grid = ConfigurationSpace(
            seed=42,
            space={
            "n_estimators": Integer("n_estimators", (20, 1500), log=True, default=100),
            "max_depth": Integer("max_depth", (1, 15), distribution=Normal(5, 3), default=3),
            "min_samples_split": Integer("min_samples_split", (2, 100), log=True, default=2),
            "min_samples_leaf": Integer("min_samples_leaf", (1, 100), log=True, default=1),
            "max_features": Categorical("max_features", [1.0, "sqrt", "log2"], default=1.0),
            "subsample": Float("subsample", (0.7, 1), default=1),
            "criterion": Categorical("criterion", ["friedman_mse", "squared_error"], default="friedman_mse"),
            "learning_rate": Float("learning_rate", (0.005, 0.3), log=True, default=0.1),
            })
        
        # workaround for now -> Problems with Normal distribution (in smac_search) (04/07/2023)
        self.smac_grid = ConfigurationSpace(
            seed=42,
            space={
            "n_estimators": Integer("n_estimators", (20, 1500), log=True, default=100),
            "max_depth": Integer("max_depth", (1, 15), default=3),
            "min_samples_split": Integer("min_samples_split", (2, 100), log=True, default=2),
            "min_samples_leaf": Integer("min_samples_leaf", (1, 100), log=True, default=1),
            "max_features": Categorical("max_features", [1.0, "sqrt", "log2"], default=1.0),
            "subsample": Float("subsample", (0.7, 1), default=1),
            "criterion": Categorical("criterion", ["friedman_mse", "squared_error"], default="friedman_mse"),
            "learning_rate": Float("learning_rate", (0.005, 0.3), log=True, default=0.1),
            })
        super().__init__(model, model_name, model_type, grid)
