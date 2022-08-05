from sklearn.naive_bayes import GaussianNB

from .main_classifier import Classifier


class GNB(Classifier):
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
        self.model_name = model_name
        self.model_type = "GNB"
        self.model = GaussianNB(**kwargs,)
        self._grid = {
            "var_smoothing": [10**i for i in range(-11, 1)]
        }
