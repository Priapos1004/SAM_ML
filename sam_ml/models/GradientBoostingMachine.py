from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer, Normal
from sklearn.ensemble import GradientBoostingClassifier

from .main_classifier import Classifier


class GBM(Classifier):
    """ GradientBoostingMachine Wrapper class """

    def __init__(
        self,
        model_name: str = "GradientBoostingMachine",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            n_estimator: number of boosting stages to perform
            criterion: function to measure the quality of a split
            max_depth: Maximum number of levels in tree
            min_samples_split: Minimum number of samples required to split a node
            min_samples_leaf: Minimum number of samples required at each leaf node
            max_features: number of features to consider when looking for the best split
            subsample: fraction of samples to be used for fitting the individual base learners
            loss: The loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss 'exponential' gradient boosting recovers the AdaBoost algorithm
            learning_rate: shrinks the contribution of each tree by learning rate

            warm_start: work with previous fit and add more estimator
            random_state: random_state for model
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
            "max_features": Categorical("max_features", ["auto", "sqrt", "log2"], default="auto"),
            "subsample": Float("subsample", (0.7, 1), default=1),
            "criterion": Categorical("criterion", ["friedman_mse", "squared_error"], default="friedman_mse"),
            "loss": Categorical("loss", ["log_loss", "deviance", "exponential"], default="log_loss"),
            "learning_rate": Float("learning_rate", (0.005, 0.3), log=True, default=0.1),
            })
        super().__init__(model, model_name, model_type, grid)
