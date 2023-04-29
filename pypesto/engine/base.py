"""Abstract engine base class."""
import abc
from typing import Any, List

from .task import Task


class Engine(abc.ABC):
    """Abstract engine base class."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def execute(
        self, tasks: List[Task], progress_bar: bool = True
    ) -> List[Any]:
        """Execute tasks.

        Parameters
        ----------
        tasks:
            List of tasks to execute.
        progress_bar:
            Whether to display a progress bar.
        """
        raise NotImplementedError("This engine is not intended to be called.")
