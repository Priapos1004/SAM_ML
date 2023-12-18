from ConfigSpace import ConfigurationSpace, Float, Integer, Normal
from xgboost import XGBRegressor

from sam_ml.config import get_n_jobs

from ..main_regressor import Regressor


class XGBR(Regressor):
    """ XGBoostRegressor Wrapper class """

    def __init__(
        self,
        model_name: str = "XGBRegressor",
        n_jobs: int = get_n_jobs(), 
        random_state: int = 42,
        **kwargs,
    ):
        """
        Parameters (important one)
        --------------------------
        random_state : int, \
                default=42
            random_state for model
        
        
        Notes
        -----
        You can use all parameters of the wrapped model when initialising the wrapper class.
        """
        model_type = "XGBR"
        model = XGBRegressor(
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )
        grid = ConfigurationSpace(
            seed=42,
            space={
            "max_depth": Integer("max_depth", (3, 10), default=6),
            "gamma": Float('gamma', (0, 9), default=0),
            'reg_alpha' : Integer('reg_alpha', (0, 180), default=0),
            'reg_lambda' : Float('reg_lambda', (0, 1), default=1),
            'colsample_bytree' : Float('colsample_bytree', (0.5, 1), default=1),
            'min_child_weight' : Integer('min_child_weight', (0, 10), default=1),
            'n_estimators': Integer("n_estimators", bounds=(50, 750), distribution=Normal(150, 100), default=100),
            "learning_rate": Float("learning_rate", bounds=(0.001, 0.30), log=True, default=0.1),
            })
        
        # workaround for now -> Problems with Normal distribution (in smac_search) (04/07/2023)
        self.smac_grid = ConfigurationSpace(
            seed=42,
            space={
            "max_depth": Integer("max_depth", (3, 10), default=6),
            "gamma": Float('gamma', (0, 9), default=0),
            'reg_alpha' : Integer('reg_alpha', (0, 180), default=0),
            'reg_lambda' : Float('reg_lambda', (0, 1), default=1),
            'colsample_bytree' : Float('colsample_bytree', (0.5, 1), default=1),
            'min_child_weight' : Integer('min_child_weight', (0, 10), default=1),
            'n_estimators': Integer("n_estimators", bounds=(50, 750), default=100),
            "learning_rate": Float("learning_rate", bounds=(0.001, 0.30), log=True, default=0.1),
            })
        super().__init__(model, model_name, model_type, grid)
