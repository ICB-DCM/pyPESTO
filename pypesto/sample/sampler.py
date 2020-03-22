import abc
from ..objective import Objective


class Sampler(abc.ABC):

    def __init__(self):
        """Constructor."""

    @abc.abstractmethod
    def sample(self, objective: Objective):
        """"Sample."""
