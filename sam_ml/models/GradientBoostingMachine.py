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
        grid = {
            "n_estimators": list(range(20, 101, 10)) + [200, 500, 1000, 1500],
            "max_depth": list(range(1, 8)) + [10, 12, 15],
            "min_samples_split": [2, 4, 6, 8, 10, 20, 40, 60, 100],
            "min_samples_leaf": [2, 4, 6, 8, 10, 20, 40, 60, 100],
            "max_features": ["auto", "sqrt", "log2", None],
            "subsample": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            "criterion": ["friedman_mse", "mse"],
            "loss": ["deviance", "exponential"],
            "learning_rate": [0.1, 0.05, 0.01, 0.005],
        }
        super().__init__(model, model_name, model_type, grid)
