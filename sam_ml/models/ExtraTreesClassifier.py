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
        grid = {
            "n_estimators": [1, 2, 4, 8, 16, 32, 64, 100, 200, 500, 1000],
            "max_depth": [2, 3, 4, 5, 6, 7, 8, 10, 15],
            "min_samples_split": [2, 3, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
        }
        super().__init__(model, model_name, model_type, grid)
