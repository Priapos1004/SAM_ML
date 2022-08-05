from sklearn.naive_bayes import BernoulliNB

from .main_classifier import Classifier


class BNB(Classifier):
    def __init__(
        self,
        model_name: str = "BernoulliNB",
        **kwargs,
    ):
        """
        @params (important one):
            binarize: threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors
            fit_prior: whether to learn class prior probabilities or not. If false, a uniform prior will be used
        """
        self.model_name = model_name
        self.model_type = "BNB"
        self.model = BernoulliNB(**kwargs,)
        self._grid = {
            "fit_prior": [True, False],
            "binarize": list(range(0, 10)),
        }
