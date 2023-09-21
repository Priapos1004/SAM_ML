from ConfigSpace import ConfigurationSpace, Integer
from sklearn.linear_model import LassoLarsCV

from ..main_regressor import Regressor


class LLCV(Regressor):
    """ LassoLarsCV Wrapper class """

    def __init__(
        self,
        model_name: str = "LassoLarsCV",
        **kwargs,
    ):
        """
        @param (important one):
            max_iter: maximum number of iterations to perform
        """
        model_type = "LLCV"
        model = LassoLarsCV(**kwargs)
        grid = ConfigurationSpace(
            seed=42,
            space={
            "max_iter": Integer("max_iter", (5, 1000), default=500),
            })

        super().__init__(model, model_name, model_type, grid)
