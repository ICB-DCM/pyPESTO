import logging
import abc
import numpy as np
from typing import Callable

from ..problem import Problem
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
            startpoint: np.ndarray,
            j_start: int,
            options: 'pypesto.OptimizeOptions',
            handle_exception: Callable):
        """
        Create the task object.

        Parameters
        ----------
        optimizer:
            The optimizer to use.
        problem:
            The problem to solve.
        startpoint:
            The point from which to start.
        j_start:
            The index of the multistart.
        options:
            Options object applying to optimization.
        handle_exception:
            Callable to apply when the optimization fails.
        """
        super().__init__()

        self.optimizer = optimizer
        self.problem = problem
        self.startpoint = startpoint
        self.j_start = j_start
        self.options = options
        self.handle_exception = handle_exception

    def execute(self) -> 'pypesto.OptimizerResult':
        logger.info(f"Executing task {self.j_start}.")
        try:
            optimizer_result = self.optimizer.minimize(
                self.problem, self.startpoint, self.j_start)
        except Exception as err:
            if self.options.allow_failed_starts:
                optimizer_result = self.handle_exception(
                    self.problem.objective, self.startpoint, self.j_start,
                    err)
            else:
                raise

        return optimizer_result
