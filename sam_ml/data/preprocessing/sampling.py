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

from .main_data import DATA

logger = setup_logger(__name__)


class Sampler(DATA):
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
        @return:
            possible values for the parameters
        """
        param = {"algorithm": ["SMOTE", "BSMOTE", "rus", "ros", "tl", "nm", "cc", "oss"]}
        return param

    def sample(self, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Function for up- and downsampling

        @return:
            tuple x_train_sampled, y_train_sampled
        """
        x_train_sampled, y_train_sampled = self.transformer.fit_resample(x_train, y_train)

        return x_train_sampled, y_train_sampled
