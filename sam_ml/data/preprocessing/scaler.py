import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from sam_ml.config import setup_logger

from .main_data import DATA

logger = setup_logger(__name__)


class Scaler(DATA):
    """ Scaler Wrapper class """

    def __init__(self, algorithm: str = "standard", **kwargs):
        """
        @param:
            algorithm: kind of scaler to use
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
        if algorithm == "standard":
            scaler = StandardScaler(**kwargs)
        elif algorithm == "minmax":
            scaler = MinMaxScaler(**kwargs)
        elif algorithm == "maxabs":
            scaler = MaxAbsScaler(**kwargs)
        elif algorithm == "robust":
            scaler = RobustScaler(**kwargs)
        elif algorithm == "normalizer":
            scaler = Normalizer(**kwargs)
        elif algorithm == "power":
            scaler = PowerTransformer(**kwargs)
        elif algorithm == "quantile":
            scaler = QuantileTransformer(**kwargs)
        elif algorithm == "quantile_normal":
            scaler = QuantileTransformer(output_distribution="normal", **kwargs)
        else:
            raise ValueError(f"algorithm='{algorithm}' is not supported")
        
        super().__init__(algorithm, scaler)

    @staticmethod
    def params() -> dict:
        """
        @return:
            possible values for the parameters of the Scaler class
        """
        param = {"scaler": ["standard", "minmax", "maxabs", "robust", "normalizer", "power", "quantile", "quantile_normal"]}
        return param

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
            scaled_ar = self.transformer.fit_transform(data)
        else:
            scaled_ar = self.transformer.transform(data)

        scaled_df = pd.DataFrame(scaled_ar, columns=columns)

        logger.debug("scaling - finished")

        return scaled_df

        