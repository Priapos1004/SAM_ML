from typing import Literal

import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import (
    ClusterCentroids,
    NearMiss,
    OneSidedSelection,
    RandomUnderSampler,
    TomekLinks,
)

from sam_ml.config import setup_logger

from ..main_data import Data

logger = setup_logger(__name__)


class Sampler(Data):
    """ sample algorithm Wrapper class """

    def __init__(self, algorithm: Literal["SMOTE", "BSMOTE", "rus", "ros", "tl", "nm", "cc", "oss"] = "ros", random_state: int = 42, sampling_strategy: str|float = "auto", **kwargs):
        """
        Parameters
        ----------
        algorithm : {"SMOTE", "BSMOTE", "rus", "ros", "tl", "nm", "cc", "oss"}, \
                defautl="ros
            which sampling algorithm to use:
            - SMOTE: Synthetic Minority Oversampling Technique (upsampling)
            - BSMOTE: BorderlineSMOTE (upsampling)
            - ros: RandomOverSampler (upsampling) (default)
            - rus: RandomUnderSampler (downsampling)
            - tl: TomekLinks (cleaning downsampling)
            - nm: NearMiss (downsampling)
            - cc: ClusterCentroids (downsampling)
            - oss: OneSidedSelection (cleaning downsampling)

        random_state : int, \
                default=42
            seed for random sampling
        sampling_strategy : str or float, \
                default="auto"
            percentage of class size of minority in relation to the class size of the majority
        \*\*kwargs:
            additional parameters for sampler
        """
        if algorithm == "SMOTE":
            sampler = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, **kwargs)
        elif algorithm == "BSMOTE":
            sampler = BorderlineSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, **kwargs)
        elif algorithm == "rus":
            sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy, **kwargs)
        elif algorithm == "ros":
            sampler = RandomOverSampler(random_state=random_state, sampling_strategy=sampling_strategy, **kwargs)
        elif algorithm == "tl":
            sampler = TomekLinks(**kwargs)
        elif algorithm == "nm":
            sampler = NearMiss(sampling_strategy=sampling_strategy, **kwargs)
        elif algorithm == "cc":
            sampler = ClusterCentroids(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
        elif algorithm == "oss":
            sampler = OneSidedSelection(random_state=random_state, **kwargs)
        else:
            raise ValueError(f"algorithm='{algorithm}' is not supported")
        
        super().__init__(algorithm, sampler)

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
        >>> from sam_ml.data.preprocessing import Sampler
        >>>
        >>> # first way without class object
        >>> params1 = Sampler.params()
        >>> print(params1)
        {"algorithm": ["ros", ...]}
        >>> # second way with class object
        >>> model = Sampler()
        >>> params2 = model.params()
        >>> print(params2)
        {"algorithm": ["ros", ...]}
        """
        param = {"algorithm": ["SMOTE", "BSMOTE", "rus", "ros", "tl", "nm", "cc", "oss"]}
        return param

    def sample(self, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Function for up- and downsampling

        Parameters
        ----------
        x_train, y_train : pd.DataFrame, pd.Series
            data to sample

        Returns
        -------
        x_train_sampled : pd.DataFrame
            sampled x data
        y_train_sampled : pd.Series
            sampled y data

        Notes
        -----
        ONLY sample the `train data`. NEVER all data because then you will have some samples in train as well as in test data with random splitting

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
        >>> # sample data
        >>> from sam_ml.data.preprocessing import Sampler
        >>>
        >>> model = Sampler()
        >>> x_train_sampled, y_train_sampled = model.sample(x_train, y_train)
        >>> print("before sampling:")
        >>> print(y_train.value_counts())
        >>> print()
        >>> print("after sampling:")
        >>> print(y_train_sampled.value_counts())
        before sampling:
        1    41
        0    40
        2    39
        Name: count, dtype: int64
        <BLANKLINE>
        after sampling:
        0    41
        1    41
        2    41
        Name: count, dtype: int64
        """
        x_train_sampled, y_train_sampled = self._transformer.fit_resample(x_train, y_train)

        return x_train_sampled, y_train_sampled
