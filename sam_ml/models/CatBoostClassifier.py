from typing import Union

from catboost import CatBoostClassifier

from .main_classifier import Classifier


class CBC(Classifier):
    def __init__(
        self,
        model_name: str = "CatBoostClassifier",
        verbose: Union[bool, int] = False,
        random_state: int = 42,
        allow_writing_files: bool = False,
        **kwargs,
    ):
        """
        @param (important one):
            depth: depth of the tree
            learning_rate: learning rate
            iterations: maximum number of trees that can be built when solving machine learning problems
            bagging_temperature: defines the settings of the Bayesian bootstrap
            random_strength: the amount of randomness to use for scoring splits when the tree structure is selected
            l2_leaf_reg: coefficient at the L2 regularization term of the cost function
            border_count: the number of splits for numerical features
        """
        self.model_name = model_name
        self.model_type = "CBC"
        self.model = CatBoostClassifier(
            verbose=verbose,
            random_state=random_state,
            allow_writing_files=allow_writing_files,
            **kwargs,
        )
        self._grid = {
            "depth": [4, 5, 6, 7, 8, 9, 10],
            "learning_rate": [0.1, 0.01, 0.02, 0.03, 0.04],
            "iterations": list(range(10, 101, 10)) + [200, 500, 1000],
            "bagging_temperature": [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0],
            "random_strength": [10**i for i in range(-9,2)],
            "l2_leaf_reg": [2, 4, 6, 8, 12, 16, 20, 24, 30],
            "border_count": [2**i for i in range(0, 9)],
        }
