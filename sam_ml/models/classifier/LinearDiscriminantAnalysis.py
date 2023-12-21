import numpy as np
from ConfigSpace import Categorical, ConfigurationSpace, Float, InCondition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..main_classifier import Classifier

np.seterr(divide = 'ignore')

class LDA(Classifier):
    """ LinearDiscriminantAnalysis Wrapper class """

    def __init__(
        self,
        model_name: str = "LinearDiscriminantAnalysis",
        **kwargs,
    ):
        """
        Parameters (important one)
        --------------------------
        solver : str,
            solver to use
        shrinkage : float or str,
            shrinkage parameters (does not work with 'svd' solver)
        
        Notes
        -----
        You can use all parameters of the wrapped model when initialising the wrapper class.
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
