import abc
<<<<<<< HEAD
import numpy as np

from ..problem import Problem
from ..objective import HistoryOptions
import pypesto


logger = logging.getLogger(__name__)
=======
>>>>>>> origin/develop


class Task(abc.ABC):
    """
    A task is one of a list of independent
    execution tasks that are submitted to the execution engine
    to be executed using the execute() method, commonly in parallel.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def execute(self):
        """
        Execute the task and return its results.
        """
<<<<<<< HEAD


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

    def execute(self) -> 'pypesto.OptimizerResult':
        logger.info(f"Executing task {self.id}.")
        optimizer_result = self.optimizer.minimize(
            problem=self.problem, x0=self.x0, id=self.id,
            allow_failed_starts=self.options.allow_failed_starts,
            history_options=self.history_options)

        return optimizer_result
=======
>>>>>>> origin/develop
