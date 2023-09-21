from .BayesianRidge import BYR
from .DecisionTreeRegressor import DTR
from .ElasticNet import EN
from .ExtraTreesRegressor import ETR
from .LassoLarsCV import LLCV
from .RandomForestRegressor import RFR
from .RegressorTest import RTest
from .SGDRegressor import SGDR

__all__ = {
    "autoML class": "RTest",
    "RandomForestRegressor": "RFR",
    "DecisionTreeRegressor": "DTR",
    "ExtraTreesRegressor": "ETR",
    "SGDRegressor": "SGDR",
    "LassoLarsCV": "LLCV",
    "ElasticNet": "EN",
    "BayesianRidge": "BYR",
}