from typing import List
from abc import abstractmethod


class Engine:
    """
    Abstract engine base class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def execute(self, tasks: List):
        raise NotImplementedError(
            "This engine is not intended to be called.")
