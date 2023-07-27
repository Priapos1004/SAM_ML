from ConfigSpace import Categorical, ConfigurationSpace, Integer
from sklearn.neighbors import KNeighborsClassifier

from .main_classifier import Classifier


class KNC(Classifier):
    """ KNeighborsClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "KNeighborsClassifier",
        **kwargs,
    ):
        """
        @param (important one):
            n_neighbors: Number of neighbors to use by default for kneighbors queries
            weights: Weight function used in prediction
            algorithm: Algorithm used to compute the nearest neighbors
            leaf_size: Leaf size passed to BallTree or KDTree
            p: number of metric that is used (manhattan, euclidean, minkowski)
            n_jobs: the number of parallel jobs to run for neighbors search [problem with n_jobs = -1 --> kernel dies]
        """
        model_type = "KNC"
        model = KNeighborsClassifier(**kwargs,)
        grid = ConfigurationSpace(
            seed=42,
            space={
            "n_neighbors": Integer("n_neighbors", (1, 30), default=5),
            "p": Integer("p", (1, 5), default=2),
            "leaf_size": Integer("leaf_size", (1, 50), default=30),
            "weights": Categorical("weights", ["uniform", "distance"], default="uniform"),
            })
        super().__init__(model, model_name, model_type, grid)
