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
            "n_estimators": Integer("n_estimators", (20, 1500), log=True),
            "max_depth": Integer("max_depth", (1, 15), distribution=Normal(4, 4)),
            "min_samples_split": Integer("min_samples_split", (2, 100), log=True),
            "min_samples_leaf": Integer("min_samples_leaf", (2, 100), log=True),
            "max_features": Categorical("max_features", ["auto", "sqrt", "log2"]),
            "subsample": Float("subsample", (0.7, 1)),
            "criterion": Categorical("criterion", ["friedman_mse", "mse"]),
            "loss": Categorical("loss", ["deviance", "exponential"]),
            "learning_rate": Float("learning_rate", (0.005, 0.1), log=True),
            })
        super().__init__(model, model_name, model_type, grid)
