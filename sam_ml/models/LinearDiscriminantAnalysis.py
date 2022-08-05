from numpy import arange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .main_classifier import Classifier


class LDA(Classifier):
    def __init__(
        self,
        model_name: str = "LinearDiscriminantAnalysis",
        **kwargs,
    ):
        """
        @param (important one):
            solver: solver to use
            shrinkage: shrinkage parameters (does not work with 'svd' solver)
        """
        self.model_name = model_name
        self.model_type = "LDA"
        self.model = LinearDiscriminantAnalysis(**kwargs)
        self._grid = {
            "solver": ["lsqr", "eigen"],
            "shrinkage": list(arange(0, 1, 0.01))+["auto"],
        }
