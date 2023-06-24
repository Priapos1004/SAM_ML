from ConfigSpace import Categorical, ConfigurationSpace, Integer
from sklearn.gaussian_process import GaussianProcessClassifier

from .main_classifier import Classifier


class GPC(Classifier):
    """ GaussianProcessClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "GaussianProcessClassifier",
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            multi_class: specifies how multi-class classification problems are handled
            max_iter_predict: the maximum number of iterations in Newton's method for approximating the posterior during predict
        """
        model_type = "GPC"
        model = GaussianProcessClassifier(
            n_jobs=n_jobs, random_state=random_state, **kwargs,
        )
        grid = ConfigurationSpace(
            seed=42,
            space={
            "multi_class": Categorical("multi_class", ["one_vs_rest", "one_vs_one"], default="one_vs_rest"),
            "max_iter_predict": Integer("max_iter_predict", (1, 1000), log=True, default=100),
            })
        super().__init__(model, model_name, model_type, grid)
