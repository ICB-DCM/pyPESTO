import logging
import abc
import numpy as np
from typing import Callable

from ..problem import Problem
from ..objective import HistoryOptions
import pypesto


logger = logging.getLogger(__name__)


class Task(abc.ABC):
    """
    A task is one of a list of independent
    execution tasks that are submitted to the execution engine
    to be executed using the execute() method, commonly in parallel.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def execute(self) -> 'pypesto.OptimizerResult':  # noqa: R0201
        """
        Execute the task and return its results.
        """


class OptimizerTask(Task):
    """
    A multistart optimization task, performed in `pypesto.minimize`.
    """

    def __init__(
            self,
            optimizer: 'pypesto.Optimizer',
            problem: Problem,
            x0: np.ndarray,
            id: str,
            options: 'pypesto.OptimizeOptions',
            history_options: HistoryOptions,
            handle_exception: Callable):
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
        handle_exception:
            Callable to apply when the optimization fails.
        """
        super().__init__()

        self.optimizer = optimizer
        self.problem = problem
        self.x0 = x0
        self.id = id
        self.options = options
        self.history_options = history_options
        self.handle_exception = handle_exception

    def execute(self) -> 'pypesto.OptimizerResult':
        logger.info(f"Executing task {self.id}.")
        try:
            optimizer_result = self.optimizer.minimize(
                problem=self.problem, x0=self.x0, id=self.id,
                history_options=self.history_options)
        except Exception as err:
            if self.options.allow_failed_starts:
                optimizer_result = self.handle_exception(
                    self.problem.objective, self.x0, self.id, err)
            else:
                raise

        return optimizer_result
