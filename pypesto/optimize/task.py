import numpy as np
import logging

from ..engine import Task
from ..objective import HistoryOptions
from ..problem import Problem
import pypesto.optimize
from contextlib import AbstractContextManager

logger = logging.getLogger(__name__)

class ConstraintRemover(AbstractContextManager):
    def __init__(self, problem, unconstrained):
        super(ConstraintRemover, self).__init__()
        self.unconstrained = unconstrained
        self.og_ub = problem.ub_full.copy()
        self.og_lb = problem.lb_full.copy()
        self.problem = problem

    def __enter__(self):
        if self.unconstrained:
            # inplace editing
            self.problem.ub_full = np.inf * np.ones(
                self.problem.ub_full.shape
            )
            self.problem.lb_full = -np.inf * np.ones(
                self.problem.lb_full.shape
            )
            return self.problem

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.problem.ub_full = self.og_ub
        self.problem.lb_full = self.og_lb

class OptimizerTask(Task):
    """
    A multistart optimization task, performed in `pypesto.minimize`.
    """

    def __init__(
            self,
            optimizer: 'pypesto.optimize.Optimizer',
            problem: Problem,
            x0: np.ndarray,
            id: str,
            options: 'pypesto.optimize.OptimizeOptions',
            history_options: HistoryOptions):
        """
        Create the task object.

        Parameters
        ----------
        optimizer:
            The optimizer to use.
        problem:
            The problem to solve.
        x0:
            The point from which to start.
        id:
            The multistart id.
        options:
            Options object applying to optimization.
        history_options:
            Optimizer history options.
        """
        super().__init__()

        self.optimizer = optimizer
        self.problem = problem
        self.x0 = x0
        self.id = id
        self.options = options
        self.history_options = history_options

    def execute(self) -> 'pypesto.optimize.OptimizerResult':
        logger.info(f"Executing task {self.id}.")

        with ConstraintRemover(self.problem,
                               self.options.unbounded_optimization) as problem:
            optimizer_result = self.optimizer.minimize(
                problem=problem, x0=self.x0, id=self.id,
                allow_failed_starts=self.options.allow_failed_starts,
                history_options=self.history_options)
        return optimizer_result
