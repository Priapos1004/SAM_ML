from ConfigSpace import Categorical, ConfigurationSpace, Float
from sklearn.svm import SVC as svc

from .main_classifier import Classifier


class SVC(Classifier):
    """ SupportVectorClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "SupportVectorClassifier",
        kernel: str = "rbf",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            random_state: random_state for model
            verbose: logging (True/False)
            C: Inverse of regularization strength
            kernel: kernel type to be used in the algorithm
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'

            class_weight: set class_weight="balanced" to deal with imbalanced data
            probability: probability=True enables probability estimates for SVM algorithms

            cache_size: Specify the size of the kernel cache (in MB)
        """
        model_type = "SVC"
        model = svc(
            kernel=kernel,
            random_state=random_state,
            **kwargs,
        )
        grid = ConfigurationSpace(
            seed=42,
            space={
            "kernel": Categorical("kernel", ["rbf", "poly", "sigmoid"], default="rbf"),
            "gamma": Float("gamma", (0.0001, 1), log=True, default=0.001),
            "C": Float("C", (0.1, 1000), log=True, default=1),
            "probability": Categorical("probability", [True, False], default=False),
            })
        super().__init__(model, model_name, model_type, grid)

    def feature_importance(self):
        if self.model.kernel == "linear":
            super().feature_importance()
        else:
            print(f"feature importance is only available for a linear kernel. You are currently using: {self.model.kernel}")
