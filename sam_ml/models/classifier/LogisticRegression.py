from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
)
from sklearn.linear_model import LogisticRegression

from ..main_classifier import Classifier


class LR(Classifier):
    """ LogisticRegression Wrapper class """

    def __init__(
        self,
        model_name: str = "LogisticRegression",
        random_state: int = 42,
        **kwargs,
    ):
        """
        Parameters (important one)
        --------------------------
        penalty : str,
            specifies the norm used in the penalization
        solver : str,
            algorithm to use in the optimization problem
        tol : float,
            tolerance for stopping criteria
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
        model_type = "LR"
        model = LogisticRegression(
            random_state=random_state,
            **kwargs,
        )
        grid = ConfigurationSpace(
            seed=42,
            space={
            "solver": Categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"], weights=[0.15, 0.15, 0.15, 0.15, 0.4], default="lbfgs"),
            "penalty": Categorical("penalty", ["l2", "elasticnet"], default="l2"),
            "C": Float("C", (0.01, 100), log=True, default=1),
            "l1_ratio": Float("l1_ratio", (0.01, 1), default=0.1),
            })
        solver_and_penalty = ForbiddenAndConjunction(
            ForbiddenEqualsClause(grid["penalty"], "elasticnet"),
            ForbiddenInClause(grid["solver"], ["newton-cg", "lbfgs", "liblinear", "sag"]),
        )
        l1_ratio_cond = EqualsCondition(grid["l1_ratio"], grid["penalty"], "elasticnet")
        grid.add_forbidden_clause(solver_and_penalty)
        grid.add_condition(l1_ratio_cond)

        super().__init__(model, model_name, model_type, grid)
