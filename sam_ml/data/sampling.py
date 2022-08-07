from typing import Union

import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler, TomekLinks
from sklearn.utils import resample


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

    def __init__(self, algorithm: str = "ros", random_state: int = 42, **kwargs):
        """
        @param:
            algorithm: which sampling algorithm to use:
                SMOTE: Synthetic Minority Oversampling Technique (upsampling)
                rus: RandomUnderSampler (downsampling)
                ros: RandomOverSampler (upsampling) (default)
                tl: TomekLinks (downsampling)
                nm: NearMiss (downsampling)
            
            random_state: seed for Random...Sampler

            **kwargs:
                additional parameters for sampler
        """
        if algorithm == "SMOTE":
            self.sampler = SMOTE(**kwargs)
        elif algorithm == "rus":
            self.sampler = RandomUnderSampler(random_state=random_state, replacement=True, **kwargs)
        elif algorithm == "ros":
            self.sampler = RandomOverSampler(random_state=random_state, **kwargs)
        elif algorithm == "tl":
            self.sampler = TomekLinks(sampling_strategy="majority", **kwargs)
        elif algorithm == "nm":
            self.sampler = NearMiss(**kwargs)
        else:
            print(f"INPUT ERROR: type='{algorithm}' does not exist --> using RandomOverSampler")
            self.sampler = RandomOverSampler(random_state=random_state, **kwargs)

    @staticmethod
    def params() -> dict:
        """
        @return:
            possible values for the parameters
        """
        param = {"algorithm": ["SMOTE", "rus", "ros", "tl", "nm"]}
        return param

    def sample(self, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Function for up- and downsampling

        @return:
            tuple x_train_sampled, y_train_sampled
        """
        x_train_sampled, y_train_sampled = self.sampler.fit_resample(x_train, y_train)

        return x_train_sampled, y_train_sampled
