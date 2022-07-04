import pandas as pd
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)


class Scaler:
    def __init__(self, scaler: str = "standard", console_out: bool = True, **kwargs):
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

            You can change all parameters of the scaler
        """
        self.console_out = console_out

        if scaler == "standard":
            if self.console_out:
                print("using StandardScaler as scaler")
            self.scaler = StandardScaler(**kwargs)

        elif scaler == "minmax":
            if self.console_out:
                print("using MinMaxScaler as scaler")
            self.scaler = MinMaxScaler(**kwargs)

        elif scaler == "maxabs":
            if self.console_out:
                print("using MaxAbsScaler as scaler")
            self.scaler = MaxAbsScaler(**kwargs)

        elif scaler == "robust":
            if self.console_out:
                print("using RobustScaler as scaler")
            self.scaler = RobustScaler(**kwargs)
            
        elif scaler == "normalizer":
            if self.console_out:
                print("using Normalizer as scaler")
            self.scaler = Normalizer(**kwargs)

        elif scaler == "power":
            if self.console_out:
                print("using PowerTransformer as scaler")
            self.scaler = PowerTransformer(**kwargs)

        elif scaler == "quantile":
            if self.console_out:
                print("using QuantileTransformer as scaler")
            self.scaler = QuantileTransformer(**kwargs)

        elif scaler == "quantile_normal":
            if self.console_out:
                print("using QuantileTransformer with output_distribution='normal' as scaler")
            self.scaler = QuantileTransformer(output_distribution="normal", **kwargs)

        else:
            print(f"Scaler '{scaler}' is no valid input -> using StandardScaler")
            self.scaler = StandardScaler()

    def scale(self, data: pd.DataFrame, train_on: bool = True) -> pd.DataFrame:
        """
        @param:
            train_on: if True, the scaler will fit_transform. Otherwise just transform

        @return:
            Dataframe with scaled data
        """
        columns = data.columns
        if self.console_out:
            print("starting to scale...")

        if train_on:
            scaled_ar = self.scaler.fit_transform(data)
        else:
            scaled_ar = self.scaler.transform(data)

        scaled_df = pd.DataFrame(scaled_ar, columns=columns)

        if self.console_out:
            print("... data scaled")

        return scaled_df

        