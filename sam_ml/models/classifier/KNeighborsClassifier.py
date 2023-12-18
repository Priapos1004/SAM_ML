from ConfigSpace import Categorical, ConfigurationSpace, Integer
from sklearn.neighbors import KNeighborsClassifier

from ..main_classifier import Classifier


class KNC(Classifier):
    """ KNeighborsClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "KNeighborsClassifier",
        **kwargs,
    ):
        """
        Parameters (important one)
        --------------------------
        n_neighbors : int,
            number of neighbors to use by default for kneighbors queries
        weights : str or callable,
            weight function used in prediction
        algorithm : str,
            algorithm used to compute the nearest neighbors
        leaf_size : int,
            leaf size passed to BallTree or KDTree
        p : int,
            number of metric that is used (manhattan, euclidean, minkowski)
        
        Notes
        -----
        You can use all parameters of the wrapped model when initialising the wrapper class.
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
