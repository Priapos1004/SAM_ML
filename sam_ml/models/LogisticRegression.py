from sklearn.linear_model import LogisticRegression

from .main_classifier import Classifier


class LR(Classifier):
    def __init__(
        self,
        model_name: str = "LogisticRegression",
        random_state: int = 42,
        **kwargs,
    ):
        """
        @param (important one):
            n_jobs: how many cores shall be used (-1 means all) (n_jobs > 1 does not have any effect when 'solver' is set to 'liblinear)
            random_state: random_state for model
            verbose: log level (higher number --> more logs)
            warm_start: work with previous fit and add more estimator
            tol: Tolerance for stopping criteria
            C: Inverse of regularization strength
            max_iter: Maximum number of iterations taken for the solvers to converge

            solver: Algorithm to use in the optimization problem
            penalty: Specify the norm of the penalty
        """
        self.model_name = model_name
        self.model_type = "LR"
        self.model = LogisticRegression(
            random_state=random_state,
            **kwargs,
        )
        self._grid = {
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag"],
            "penalty": ["l2"],
            "C": [100, 10, 1.0, 0.1, 0.01],
        }
