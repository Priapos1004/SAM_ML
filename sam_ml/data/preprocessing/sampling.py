from typing import Union

import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import (
    ClusterCentroids,
    NearMiss,
    OneSidedSelection,
    RandomUnderSampler,
    TomekLinks,
)
from sklearn.utils import resample

from sam_ml.config import setup_logger

logger = setup_logger(__name__)


def simple_upsample(x_train: pd.DataFrame, y_train: pd.Series, label: Union[int, str] = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    function written by Sughu

    @param:
        x_train, y_train: trainings data for upsamling
        label: label that shall be upsampled
    
    @return:
        tuple x_train, y_train
    """
    count_per_sample = max(y_train.value_counts())
    # Reset indexes
    x_train.reset_index(drop=True)
    y_train.reset_index(drop=True)
    y_train = pd.DataFrame(y_train)
    y_train.columns = ["y"]

    # Stacking horizontally
    df = pd.concat([x_train, y_train], axis=1)

    df_1 = df[df["y"] == label]
    # set other classes to another dataframe
    other_df = df[df["y"] != label]
    
    # upsample the minority class
    df_1_upsampled = resample(
        df_1, random_state=42, n_samples=count_per_sample, replace=True
    )
    
    # concatenate the upsampld dataframe
    df_1_upsampled = df_1_upsampled.reset_index(drop=True)
    other_df = other_df.reset_index(drop=True)

    df_upsampled = pd.concat([df_1_upsampled, other_df])

    # Split into X and y
    x_train = df_upsampled.iloc[:, :-1]
    y_train = df_upsampled.iloc[:, -1]

    return x_train, y_train

class Sampler:
    """ sample algorithm Wrapper class """

    def __init__(self, algorithm: str = "ros", random_state: int = 42, sampling_strategy="auto", **kwargs):
        """
        @param:
            algorithm: which sampling algorithm to use:
                SMOTE: Synthetic Minority Oversampling Technique (upsampling)
                BSMOTE: BorderlineSMOTE (upsampling)
                ros: RandomOverSampler (upsampling) (default)
                rus: RandomUnderSampler (downsampling)
                tl: TomekLinks (cleaning downsampling)
                nm: NearMiss (downsampling)
                cc: ClusterCentroids (downsampling)
                oss: OneSidedSelection (cleaning downsampling)
            
            random_state: seed for random sampling

            sampling_strategy: percentage of minority class size of majority class size

            **kwargs:
                additional parameters for sampler
        """
        self.algorithm = algorithm
        self._grid: dict[str, list] = {} # for pipeline structure

        if algorithm == "SMOTE":
            self.sampler = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, **kwargs)
        elif algorithm == "BSMOTE":
            self.sampler = BorderlineSMOTE(random_state=random_state, sampling_strategy=sampling_strategy, **kwargs)
        elif algorithm == "rus":
            self.sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy, **kwargs)
        elif algorithm == "ros":
            self.sampler = RandomOverSampler(random_state=random_state, sampling_strategy=sampling_strategy, **kwargs)
        elif algorithm == "tl":
            self.sampler = TomekLinks(**kwargs)
        elif algorithm == "nm":
            self.sampler = NearMiss(sampling_strategy=sampling_strategy, **kwargs)
        elif algorithm == "cc":
            self.sampler = ClusterCentroids(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
        elif algorithm == "oss":
            self.sampler = OneSidedSelection(random_state=random_state, **kwargs)
        else:
            raise ValueError(f"type='{algorithm}' is not supported")

    def __repr__(self) -> str:
        sampler_params: str = ""
        param_dict = self.get_params(False)
        for key in param_dict:
            if type(param_dict[key]) == str:
                sampler_params += key+"='"+str(param_dict[key])+"', "
            else:
                sampler_params += key+"="+str(param_dict[key])+", "
        return f"Sampler({sampler_params})"

    @staticmethod
    def params() -> dict:
        """
        @return:
            possible values for the parameters
        """
        param = {"algorithm": ["SMOTE", "BSMOTE", "rus", "ros", "tl", "nm", "cc", "oss"]}
        return param

    def get_params(self, deep: bool = True):
        return {"algorithm": self.algorithm} | self.sampler.get_params(deep)

    def set_params(self, **params):
        self.sampler.set_params(**params)
        return self

    def sample(self, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Function for up- and downsampling

        @return:
            tuple x_train_sampled, y_train_sampled
        """
        x_train_sampled, y_train_sampled = self.sampler.fit_resample(x_train, y_train)

        return x_train_sampled, y_train_sampled
