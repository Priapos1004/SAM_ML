from ConfigSpace import Categorical, ConfigurationSpace, Float, InCondition
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
        grid = ConfigurationSpace(
            seed=42,
            space={
            "solver": Categorical("solver", ["lsqr", "eigen", "svd"], weights=[0.475, 0.475, 0.05], default="svd"),
            "shrinkage": Float("shrinkage", (0, 1), default=0),
            })
        shrinkage_cond = InCondition(grid["shrinkage"], grid["solver"], ["lsqr", "eigen"])
        grid.add_condition(shrinkage_cond)
        super().__init__(model, model_name, model_type, grid)
