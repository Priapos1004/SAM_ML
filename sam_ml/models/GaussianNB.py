from ConfigSpace import ConfigurationSpace, Float
from sklearn.naive_bayes import GaussianNB

from .main_classifier import Classifier


class GNB(Classifier):
    """ GaussianNB Wrapper class """

    def __init__(
        self,
        model_name: str = "GaussianNB",
        **kwargs,
    ):
        """
        @params:
            priors: Prior probabilities of the classes. If specified the priors are not adjusted according to the data
            var_smoothing: Portion of the largest variance of all features that is added to variances for calculation stability
        """
        model_type = "GNB"
        model = GaussianNB(**kwargs,)
        grid = ConfigurationSpace(
            seed=42,
            space={
            "var_smoothing": Float("var_smoothing", (1e-11, 1), log=True, default=1e-9)
            })
        super().__init__(model, model_name, model_type, grid)
