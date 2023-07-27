from ConfigSpace import Categorical, ConfigurationSpace, Integer
from sklearn.naive_bayes import BernoulliNB

from .main_classifier import Classifier


class BNB(Classifier):
    """ BernoulliNB Wrapper class """

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
        model_type = "BNB"
        model = BernoulliNB(**kwargs,)
        grid = ConfigurationSpace(
            seed=42,
            space={
            "fit_prior": Categorical("fit_prior", [True, False], default=True),
            "binarize": Integer("binarize", (0, 10), default=0),
            })
        super().__init__(model, model_name, model_type, grid)
