from typing import List
import abc

from .task import Task


class Engine(abc.ABC):
    """
    Abstract engine base class.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def execute(self, tasks: List[Task]):
        """Execute tasks.

        Parameters
        ----------
        tasks:
            List of tasks to execute.
        """
        raise NotImplementedError(
            "This engine is not intended to be called.")
