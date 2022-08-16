import warnings

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
        grid = {
            "reg_param": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        }
        super().__init__(model, model_name, model_type, grid)
