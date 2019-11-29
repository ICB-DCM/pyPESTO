from typing import List
import abc


class Engine(abc.ABC):
    """
    Abstract engine base class.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def execute(self, tasks: List):
        raise NotImplementedError(
            "This engine is not intended to be called.")
