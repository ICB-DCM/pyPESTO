import abc
import numpy as np
from typing import Any, Dict, Tuple, Union

from ..objective import Objective
from ..problem import Problem
from .result import McmcPtResult


class Sampler(abc.ABC):
    """Sampler base class, not functional on its own.

    The sampler maintains an internal chain, which is initialized in
    `initialize`, and updated in `sample`.
    """

    @abc.abstractmethod
    def initialize(self, problem: Problem, x0: np.ndarray):
        """Initialize the sampler.

        Parameters
        ----------
        problem:
            The problem for which to sample.
        x0:
            Should, but is not required to, be used as initial parameter.
        """

    @abc.abstractmethod
    def sample(
            self, n_samples: int, beta: float = 1.
    ):
        """Perform sampling.

        Parameters
        ----------
        n_samples:
            Number of samples to generate.
        beta:
            Inverse of the temperature to which the system is elevated.
        """

    @abc.abstractmethod
    def get_samples(self) -> McmcPtResult:
        """Get the generated samples."""


class InternalSample:

    def __init__(self, x: np.ndarray, llh: float):
        self.x = x
        self.llh = llh


class TemperableSampler(Sampler):
    """Sampler to be used inside a parallel tempering sampler.

    The last sample can be obtained via `get_last_sample` and set via
    `set_last_sample`.
    """

    @abc.abstractmethod
    def get_last_sample(self) -> InternalSample:
        """Get the last sample in the chain.

        Returns
        -------
        """

    @abc.abstractmethod
    def set_last_sample(self, sample: InternalSample):
        """
        Set the last sample in the chain to the passed value.

        Parameters
        ----------
        sample:
            The sample that will replace the last sample in the chain.
        """
