import warnings

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from .main_classifier import Classifier

warnings.filterwarnings("ignore", category=UserWarning)


class QDA(Classifier):
    def __init__(
        self,
        model_name: str = "QuadraticDiscriminantAnalysis",
        **kwargs,
    ):
        """
        @param (important one):
            reg_param: regularizes the per-class covariance estimates by transforming
        """
        self.model_name = model_name
        self.model_type = "QDA"
        self.model = QuadraticDiscriminantAnalysis(**kwargs)
        self._grid = {
            "reg_param": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        }
