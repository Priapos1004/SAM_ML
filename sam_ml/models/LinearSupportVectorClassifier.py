from sklearn.svm import LinearSVC

from .main_classifier import Classifier


class LSVC(Classifier):
    """ LinearSupportVectorClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "LinearSupportVectorClassifier",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            random_state: random_state for model
            verbose: logging
            penalty: specifies the norm used in the penalization
            dual: select the algorithm to either solve the dual or primal optimization problem
            C: Inverse of regularization strength
            max_iter: Maximum number of iterations taken for the solvers to converge
        """
        model_type = "LSVC"
        model = LinearSVC(
            random_state=random_state,
            **kwargs,
        )
        grid = {
            "penalty": ["l1", "l2"],
            "dual": [True, False],
            "C": [10**i for i in range(-5, 6)]
        }
        super().__init__(model, model_name, model_type, grid)
