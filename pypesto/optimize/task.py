import numpy as np
import logging

from ..engine import Task
from ..objective import HistoryOptions
from ..problem import Problem
import pypesto.optimize

logger = logging.getLogger(__name__)


class OptimizerTask(Task):
    """A multistart optimization task, performed in `pypesto.minimize`."""

    def __init__(
            self,
            optimizer: 'pypesto.optimize.Optimizer',
            problem: Problem,
            x0: np.ndarray,
            id: str,
            options: 'pypesto.optimize.OptimizeOptions',
            history_options: HistoryOptions,
            report_hess: bool,
            report_sres: bool,
    ):
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
        report_hess
        """
        super().__init__()

        self.optimizer = optimizer
        self.problem = problem
        self.x0 = x0
        self.id = id
        self.options = options
        self.history_options = history_options
        self.report_hess = report_hess
        self.report_sres = report_sres

    def execute(self) -> 'pypesto.optimize.OptimizerResult':
        """Execute the task."""
        logger.info(f"Executing task {self.id}.")

        optimizer_result = self.optimizer.minimize(
            problem=self.problem, x0=self.x0, id=self.id,
            allow_failed_starts=self.options.allow_failed_starts,
            history_options=self.history_options)
        if not self.report_hess:
            optimizer_result.hess = None
        if not self.report_sres:
            optimizer_result.sres = None
        return optimizer_result
