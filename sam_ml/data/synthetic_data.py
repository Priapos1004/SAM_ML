import random

import pandas as pd
from sklearn.datasets import make_classification


def rescale(col: pd.Series, range_low: int, range_high: int) -> pd.Series:
    """
    Function to rescale values between any given range
    """
    min_val = min(col)
    max_val = max(col)
    scaled = (range_high - range_low) * ((col-min_val)/(max_val - min_val)) + range_low
    return scaled.apply(round)

def synt_data(columns: dict[str, list], label_name: str = "label", sample_num: int = 1000, classes_num: int = 2, weights: list[float] = None, class_sep: float = 0.98, random_seed: int = 42) -> pd.DataFrame:
    """
    @param:
        columns -> dict {"column_name1": [informativ: bool, range: tuple]}
            inforamtiv: is the feature important?
            range: (min_value, max_value) of column
    """

    total_feature_num = len(columns)
    informativ_features = [i for i in list(columns.keys()) if columns[i][0]==True]
    random_features = [i for i in list(columns.keys()) if columns[i][0]==False]

    X, y = make_classification(n_samples=sample_num, n_features=total_feature_num, n_informative = len(informativ_features), n_redundant=0, n_repeated=0,
                            class_sep=class_sep, random_state=random_seed, weights = weights, n_classes=classes_num)

    # Adding column names
    X = pd.DataFrame(X, columns =informativ_features+random_features)

    # Rescaling features to a believable range. For instance, age between 25 and 55 
    for i in informativ_features:
        X[i] = rescale(X[i],columns[i][1][0], columns[i][1][1])

    for i in random_features:
        random.seed(random_seed)
        X[i] = [random.randint(columns[i][1][0], columns[i][1][1]) for _ in range(sample_num)]

    data = pd.concat([pd.DataFrame(X),pd.Series(y, name=label_name)], axis=1)

    #data.to_csv("synthetic_attrition_dataset.csv", index=False)
    return data
