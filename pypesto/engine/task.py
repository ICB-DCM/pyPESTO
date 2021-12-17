"""Abstract Task class."""
import abc


class Task(abc.ABC):
    """
    Abstract Task class.

    A task is one of a list of independent execution tasks that are
    submitted to the execution engine to be executed using the execute()
    method, commonly in parallel.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def execute(self):
        """Execute the task and return its results."""
