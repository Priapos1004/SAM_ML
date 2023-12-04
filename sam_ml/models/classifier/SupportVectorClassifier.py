from ConfigSpace import Categorical, ConfigurationSpace, Float
from sklearn.svm import SVC as svc

from ..main_classifier import Classifier


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
        Parameters (important one)
        --------------------------
        C : float,
            inverse of regularization strength
        kernel : str,
            kernel type to be used in the algorithm
        gamma : str or float,
            kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        class_weight : dict or str,
            set class_weight="balanced" to deal with imbalanced data
        probability : bool,
            probability=True enables probability estimates for SVM algorithms
        random_state : int, \
                default=42
            random_state for model
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
        """
        Function to generate a matplotlib plot of the top45 feature importance from the model. 
        You can only use the method if you trained your model before.

        Returns
        -------
        plt.show object
        """
        if self.model.kernel == "linear":
            super().feature_importance()
        else:
            raise ValueError(f"feature importance is only available for a linear kernel. You are currently using: {self.model.kernel}")
