"""Abstract engine base class."""
import abc
from typing import Any

from .task import Task


class Engine(abc.ABC):
    """Abstract engine base class."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def execute(
        self, tasks: list[Task], progress_bar: bool = True
    ) -> list[Any]:
        """Execute tasks.

        Parameters
        ----------
        tasks:
            List of tasks to execute.
        progress_bar:
            Whether to display a progress bar. Defaults to ``True``.
        """
        raise NotImplementedError("This engine is not intended to be called.")
