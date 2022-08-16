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
        grid = {
            "multi_class": ["one_vs_rest", "one_vs_one"],
            "max_iter_predict": [1, 10, 50, 100, 200, 500, 1000],
        }
        super().__init__(model, model_name, model_type, grid)
