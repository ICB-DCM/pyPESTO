import abc
import numpy as np
from typing import Any, Tuple, Union

from ..objective import Objective
from ..problem import Problem
from .result import McmcPtResult


class Sampler(abc.ABC):
    """Sampler base class, not functional on its own.
    """

    @abc.abstractmethod
    def sample(
            self, problem: Problem, x0: np.ndarray = None
    ) -> Union[McmcPtResult, Any]:
        """"Perform sampling.

        Parameters
        ----------
        problem:
            The problem for which to sample.
        x0:
            Initial parameter.

        Returns
        -------
        sample_result:
            The sampling results in standardized format.
        """


class InternalSampler:
    """Sampler to be used inside a parallel tempering sampler."""

    @abc.abstractmethod
    def perform_step(
            self, x: np.ndarray, llh: float, objective: Objective
    ) -> Tuple[np.ndarray, float]:
        """
        Perform a step.

        Parameters
        ----------
        x:
            The current parameter.
        llh:
            The current log-likelihood value.
        objective:
            Objective log
        """
