from numpy import arange
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .main_classifier import Classifier


class LDA(Classifier):
    """ LinearDiscriminantAnalysis Wrapper class """

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
        model_type = "LDA"
        model = LinearDiscriminantAnalysis(**kwargs)
        grid = {
            "solver": ["lsqr", "eigen"],
            "shrinkage": list(arange(0, 1, 0.01))+["auto"],
        }
        super().__init__(model, model_name, model_type, grid)
