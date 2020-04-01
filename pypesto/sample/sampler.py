import abc
from typing import Any, Union

from ..problem import Problem
from .result import McmcPtResult


class Sampler(abc.ABC):
    """Sampler base class, not functional on its own.
    """

    @abc.abstractmethod
    def sample(self, problem: Problem) -> Union[McmcPtResult, Any]:
        """"Perform sampling.

        Parameters
        ----------
        problem:
            The problem for which to sample.

        Returns
        -------
        sample_result:
            The sampling results in standardized format.
        """
