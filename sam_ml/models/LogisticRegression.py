import numpy as np
from ConfigSpace import (Categorical, ConfigurationSpace, EqualsCondition,
                         Float, ForbiddenAndConjunction, ForbiddenEqualsClause,
                         ForbiddenInClause)
from sklearn.linear_model import LogisticRegression

from .main_classifier import Classifier


class LR(Classifier):
    """ LogisticRegression Wrapper class """

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
