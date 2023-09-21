from ConfigSpace import Categorical, ConfigurationSpace, EqualsCondition, Float
from sklearn.linear_model import SGDRegressor

from ..main_regressor import Regressor


class SGDR(Regressor):
    """ SGDRegressor Wrapper class """

    def __init__(
        self,
        model_name: str = "SGDRegressor",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            penalty: Specify the norm of the penalty
            alpha: Constant that multiplies the regularization term (the higher the value, the stronger the regularization)
        """
        model_type = "SGDR"
        model = SGDRegressor(random_state=random_state, **kwargs)
        grid = ConfigurationSpace(
            seed=42,
            space={
            "penalty": Categorical("penalty", ["l2", "elasticnet"], default="l2"),
            "alpha": Float("alpha", (0.0001, 100), log=True, default=0.0001),
            "l1_ratio": Float("l1_ratio", (0.01, 1), default=0.15),
            "learning_rate": Categorical("learning_rate", ["constant", "optimal", "invscaling", "adaptive"], default="invscaling")
            })
        l1_ratio_cond = EqualsCondition(grid["l1_ratio"], grid["penalty"], "elasticnet")
        grid.add_condition(l1_ratio_cond)

        super().__init__(model, model_name, model_type, grid)
