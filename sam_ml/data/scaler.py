import pandas as pd
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)

from sam_ml.config import setup_logger

logger = setup_logger(__name__)


class Scaler:
    """ Scaler Wrapper class """

    def __init__(self, scaler: str = "standard", **kwargs):
        """
        @param:
            scaler: kind of scaler to use
                'standard': StandardScaler
                'minmax': MinMaxScaler
                'maxabs': MaxAbsScaler
                'robust': RobustScaler
                'normalizer': Normalizer
                'power': PowerTransformer with method="yeo-johnson"
                'quantile': QuantileTransformer (default of QuantileTransformer)
                'quantile_normal': QuantileTransformer with output_distribution="normal" (gaussian pdf)

            **kwargs:
                additional parameters for scaler
        """
        self.scaler_type = scaler
        self._grid: dict[str, list] = {} # for pipeline structure

        if scaler == "standard":
            logger.debug("using StandardScaler as scaler")
            self.scaler = StandardScaler(**kwargs)

        elif scaler == "minmax":
            logger.debug("using MinMaxScaler as scaler")
            self.scaler = MinMaxScaler(**kwargs)

        elif scaler == "maxabs":
            logger.debug("using MaxAbsScaler as scaler")
            self.scaler = MaxAbsScaler(**kwargs)

        elif scaler == "robust":
            logger.debug("using RobustScaler as scaler")
            self.scaler = RobustScaler(**kwargs)
            
        elif scaler == "normalizer":
            logger.debug("using Normalizer as scaler")
            self.scaler = Normalizer(**kwargs)

        elif scaler == "power":
            logger.debug("using PowerTransformer as scaler")
            self.scaler = PowerTransformer(**kwargs)

        elif scaler == "quantile":
            logger.debug("using QuantileTransformer as scaler")
            self.scaler = QuantileTransformer(**kwargs)

        elif scaler == "quantile_normal":
            logger.debug("using QuantileTransformer with output_distribution='normal' as scaler")
            self.scaler = QuantileTransformer(output_distribution="normal", **kwargs)

        else:
            logger.error(f"scaler='{scaler}' is no valid input -> using StandardScaler")
            self.scaler = StandardScaler()
            self.scaler_type = "standard"

    def __repr__(self) -> str:
        scaler_params: str = ""
        param_dict = self.get_params(False)
        for key in param_dict:
            if type(param_dict[key]) == str:
                scaler_params += key+"='"+str(param_dict[key])+"', "
            else:
                scaler_params += key+"="+str(param_dict[key])+", "
        return f"Scaler({scaler_params})"

    @staticmethod
    def params() -> dict:
        """
        @return:
            possible values for the parameters of the Scaler class
        """
        param = {"scaler": ["standard", "minmax", "maxabs", "robust", "normalizer", "power", "quantile", "quantile_normal"]}
        return param

    def get_params(self, deep: bool = True):
        return {"scaler": self.scaler_type} | self.scaler.get_params(deep)

    def set_params(self, **params):
        self.scaler.set_params(**params)
        return self

    def scale(self, data: pd.DataFrame, train_on: bool = True) -> pd.DataFrame:
        """
        @param:
            train_on: if True, the scaler will fit_transform. Otherwise just transform

        @return:
            Dataframe with scaled data
        """
        columns = data.columns
        logger.debug("scaling - started")

        if train_on:
            scaled_ar = self.scaler.fit_transform(data)
        else:
            scaled_ar = self.scaler.transform(data)

        scaled_df = pd.DataFrame(scaled_ar, columns=columns)

        logger.debug("scaling - finished")

        return scaled_df

        