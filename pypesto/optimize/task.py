import logging

import numpy as np

import pypesto.optimize

from ..engine import Task
from ..history import HistoryOptions
from ..problem import Problem
from ..result import OptimizerResult

logger = logging.getLogger(__name__)


class OptimizerTask(Task):
    """A multistart optimization task, performed in `pypesto.minimize`."""

    def __init__(
        self,
        optimizer: "pypesto.optimize.Optimizer",
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions,
        optimize_options: "pypesto.optimize.OptimizeOptions",
    ):
        """Create the task object.

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
        self.optimize_options = optimize_options
        self.history_options = history_options

    def execute(self) -> OptimizerResult:
        """Execute the task."""
        logger.debug(f"Executing task {self.id}.")
        # check for supplied x_guess support
        self.optimizer.check_x0_support(self.problem.x_guesses)

        optimizer_result = self.optimizer.minimize(
            problem=self.problem,
            x0=self.x0,
            id=self.id,
            history_options=self.history_options,
            optimize_options=self.optimize_options,
        )

        if not self.optimize_options.report_hess:
            optimizer_result.hess = None
        if not self.optimize_options.report_sres:
            optimizer_result.sres = None
        return optimizer_result
