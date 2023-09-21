from ConfigSpace import Categorical, ConfigurationSpace, Float
from sklearn.linear_model import BayesianRidge

from ..main_regressor import Regressor


class BYR(Regressor):
    """ BayesianRidge Wrapper class """

    def __init__(
        self,
        model_name: str = "BayesianRidge",
        **kwargs,
    ):
        """
        @param (important one):
            alpha_init: initial value for alpha (precision of the noise)
            lambda_init: initial value for lambda (precision of the weights)
        """
        model_type = "BYR"
        model = BayesianRidge(**kwargs)
        grid = ConfigurationSpace(
            seed=42,
            space={
            "alpha_init": Float("alpha_init", (1, 1.9), default=1),
            "lambda_init": Float("lambda_init", (1e-09, 1), log=True, default=1),
            })

        super().__init__(model, model_name, model_type, grid)
