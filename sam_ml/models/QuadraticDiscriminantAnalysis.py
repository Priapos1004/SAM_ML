import warnings

from ConfigSpace import ConfigurationSpace, Float
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from .main_classifier import Classifier

warnings.filterwarnings("ignore", category=UserWarning)


class QDA(Classifier):
    """ QuadraticDiscriminantAnalysis Wrapper class """

    def __init__(
        self,
        model_name: str = "QuadraticDiscriminantAnalysis",
        **kwargs,
    ):
        """
        @param (important one):
            reg_param: regularizes the per-class covariance estimates by transforming
        """
        model_type = "QDA"
        model = QuadraticDiscriminantAnalysis(**kwargs)
        grid = ConfigurationSpace(
            seed=42,
            space={
            "reg_param": Float("reg_param", (0, 1), default=0),
            })
        super().__init__(model, model_name, model_type, grid)
