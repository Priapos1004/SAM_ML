from typing import Literal

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

from ..main_data import Data

logger = setup_logger(__name__)


class Scaler(Data):
    """ Scaler Wrapper class """

    def __init__(self, algorithm: Literal["standard", "minmax", "maxabs", "robust", "normalizer", "power", "quantile", "quantile_normal"] = "standard", **kwargs):
        """
        Parameters
        ----------
        algorithm : {"standard", "minmax", "maxabs", "robust", "normalizer", "power", "quantile", "quantile_normal"}, \
                default="standard"
            which scaling algorithm to use:
            - 'standard': StandardScaler
            - 'minmax': MinMaxScaler
            - 'maxabs': MaxAbsScaler
            - 'robust': RobustScaler
            - 'normalizer': Normalizer
            - 'power': PowerTransformer with method="yeo-johnson"
            - 'quantile': QuantileTransformer (default of QuantileTransformer)
            - 'quantile_normal': QuantileTransformer with output_distribution="normal" (gaussian pdf)

        \*\*kwargs:
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
        Function to get the possible parameter values for the class

        Returns
        -------
        param : dict
            possible values for the parameter "algorithm"

        Examples
        --------
        >>> # get possible parameters
        >>> from sam_ml.data.preprocessing import Scaler
        >>>
        >>> # first way without class object
        >>> params1 = Scaler.params()
        >>> print(params1)
        {"algorithm": ["standard", ...]}
        >>> # second way with class object
        >>> model = Scaler()
        >>> params2 = model.params()
        >>> print(params2)
        {"algorithm": ["standard", ...]}
        """
        param = {"algorithm": ["standard", "minmax", "maxabs", "robust", "normalizer", "power", "quantile", "quantile_normal"]}
        return param

    def scale(self, data: pd.DataFrame, train_on: bool = True) -> pd.DataFrame:
        """
        Function to scale/normalise data

        Parameters
        ----------
        data : pd.DataFrame
            data that shall be scaled
        train_on : bool, \
                defautl=True
            If ``True``, the estimator instance will fit_transform. Otherwise, just transform

        Returns
        -------
        scaled_df : pd.DataFrame
            Dataframe with scaled data

        Examples
        --------
        >>> # load data (replace with own data)
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> df = load_iris()
        >>> X, y = pd.DataFrame(df.data, columns=df.feature_names), pd.Series(df.target)
        >>> x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.80, random_state=42)
        >>>
        >>> # scale data
        >>> from sam_ml.data.preprocessing import Scaler
        >>>
        >>> model = Scaler()
        >>> x_train_scaled = model.scale(x_train) # train scaler
        >>> x_test_scaled = model.scale(x_test, train_on=False) # scale test data
        >>> print("before scaling:")
        >>> print(x_train.iloc[0])
        >>> print()
        >>> print("after scaling:")
        >>> print(x_train_scaled.iloc[0])
        before scaling:
        sepal length (cm)    4.6
        sepal width (cm)     3.6
        petal length (cm)    1.0
        petal width (cm)     0.2
        Name: 22, dtype: float64
        <BLANKLINE>
        after scaling:
        sepal length (cm)   -1.473937
        sepal width (cm)     1.203658
        petal length (cm)   -1.562535
        petal width (cm)    -1.312603
        Name: 0, dtype: float64
        """
        columns = data.columns
        logger.debug("scaling - started")

        if train_on:
            scaled_ar = self._transformer.fit_transform(data)
        else:
            scaled_ar = self._transformer.transform(data)

        scaled_df = pd.DataFrame(scaled_ar, columns=columns)

        logger.debug("scaling - finished")

        return scaled_df

        