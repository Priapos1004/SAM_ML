from ConfigSpace import Categorical, ConfigurationSpace, Float
from sklearn.linear_model import ElasticNet

from ..main_regressor import Regressor


class EN(Regressor):
    """ ElasticNet Wrapper class """

    def __init__(
        self,
        model_name: str = "ElasticNet",
        **kwargs,
    ):
        """
        @param (important one):
            selection: if set to 'random', a random coefficient is updated every iteration rather than looping over features sequentially by default. This (setting to 'random') often leads to significantly faster convergence especially when tol is higher than 1e-4
        """
        model_type = "EN"
        model = ElasticNet(**kwargs)
        grid = ConfigurationSpace(
            seed=42,
            space={
            "l1_ratio": Float("l1_ratio", (0.1, 0.9), default=0.5),
            "tol": Float("tol", (1e-06, 1e-02), log=True, default=0.0001),
            "selection": Categorical("selection", ["cyclic", "random"], default="cyclic"),
            })

        super().__init__(model, model_name, model_type, grid)
