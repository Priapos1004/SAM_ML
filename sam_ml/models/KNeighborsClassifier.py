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
        grid = {
            "n_neighbors": list(range(1,30)),
            "p": [1, 2, 3, 4, 5],
            "leaf_size": list(range(1,50)),
            "weights": ["uniform", "distance"],
        }
        super().__init__(model, model_name, model_type, grid)
