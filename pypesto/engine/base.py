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
    def execute(self, tasks: List[Task], progress_bar: bool = True):
        """Execute tasks.

        Parameters
        ----------
        tasks:
            List of tasks to execute.
        progress_bar:
            Indicates, whether a progress bar should be displayed.
            Default is True.
        """
        raise NotImplementedError(
            "This engine is not intended to be called.")
