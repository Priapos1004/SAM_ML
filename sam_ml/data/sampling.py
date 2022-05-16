from typing import Union

import pandas as pd
from sklearn.utils import resample


def upsample(x_train: pd.DataFrame, y_train: pd.DataFrame, label: Union[int, str] = 1) -> tuple[pd.DataFrame]:
    """
    @param:
        x_train, y_train - trainings data for upsamling
        label - label that shall be upsampled
    
    @return:
        tuple with x_train, y_train
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