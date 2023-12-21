from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
)
from sklearn.svm import LinearSVC

from ..main_classifier import Classifier


class LSVC(Classifier):
    """ LinearSupportVectorClassifier Wrapper class """

    def __init__(
        self,
        model_name: str = "LinearSupportVectorClassifier",
        random_state: int = 42,
        **kwargs,
    ):
        """
        Parameters (important one)
        --------------------------
        penalty : str,
            specifies the norm used in the penalization
        dual : bool,
            select the algorithm to either solve the dual or primal optimization problem
        C : float,
            inverse of regularization strength
        max_iter : int,
            maximum number of iterations taken for the solvers to converge
        random_state : int, \
                default=42
            random_state for model
        
        Notes
        -----
        You can use all parameters of the wrapped model when initialising the wrapper class.
        """
        model_type = "LSVC"
        model = LinearSVC(
            random_state=random_state,
            **kwargs,
        )
        grid = ConfigurationSpace(
            seed=42,
            space={
            "penalty": Categorical("penalty", ["l1", "l2"], default="l2"),
            "dual": Categorical("dual", [True, False], default=True),
            "C": Float("C", (0.1, 1000), log=True, default=1),
            })
        penalty_dual = ForbiddenAndConjunction(
            ForbiddenEqualsClause(grid["dual"], True),
            ForbiddenEqualsClause(grid["penalty"], "l1"),
        )
        grid.add_forbidden_clause(penalty_dual)
        super().__init__(model, model_name, model_type, grid)
