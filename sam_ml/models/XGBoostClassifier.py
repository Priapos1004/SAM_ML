from ConfigSpace import ConfigurationSpace, Float, Integer, Normal
from xgboost import XGBClassifier

from .main_classifier import Classifier


class XGBC(Classifier):
    """ SupportVectorClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "XGBClassifier",
        n_jobs: str = -1,
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            random_state: random_state for model
            n_jobs: how many cores shall be used (-1 means all)
        """
        model_type = "XGBC"
        model = XGBClassifier(
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
        super().__init__(model, model_name, model_type, grid)

    def feature_importance(self):
        super().feature_importance()
