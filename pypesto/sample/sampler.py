import abc
import numpy as np
from typing import Dict, List, Union

from ..problem import Problem
from .result import McmcPtResult


class Sampler(abc.ABC):
    """Sampler base class, not functional on its own.

    The sampler maintains an internal chain, which is initialized in
    `initialize`, and updated in `sample`.
    """

    def __init__(self, options: Dict = None):
        self.options = self.__class__.translate_options(options)

    @abc.abstractmethod
    def initialize(self,
                   problem: Problem,
                   x0: Union[np.ndarray, List[np.ndarray]]):
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
            self, n_samples: int, beta: float = 1.,
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

    @classmethod
    def default_options(cls) -> Dict:
        """Convenience method to set/get default options.

        Returns
        -------
        default_options:
            Default sampler options.
        """
        return {}

    @classmethod
    def translate_options(cls, options):
        """Convenience method to translate options and fill in defaults.

        Parameters
        ----------
        options:
            Options configuring the sampler.
        """
        used_options = cls.default_options()
        if options is None:
            options = {}
        for key, val in options.items():
            if key not in used_options:
                raise KeyError(f"Cannot handle key {key}.")
            used_options[key] = val
        return used_options


class InternalSample:
    """
    This is the exchange object provided and accepted by
    `InternalSampler.get_last_sample()`, `InternalSampler.set_last_sample()`.
    It carries all information needed to check whether to swap between chains,
    and to continue the chain from the updated sample.

    Attributes
    ----------
    x:
        Parameter values.
    lpost:
        Log-posterior value (negative function value).
    lprior:
        Log-prior value (negative function value).
    """

    def __init__(self, x: np.ndarray, lpost: float, lprior: float):
        self.x = x
        self.lpost = lpost
        self.lprior = lprior


class InternalSampler(Sampler):
    """Sampler to be used inside a parallel tempering sampler.

    The last sample can be obtained via `get_last_sample` and set via
    `set_last_sample`.
    """

    @abc.abstractmethod
    def get_last_sample(self) -> InternalSample:
        """Get the last sample in the chain.

        Returns
        -------
        internal_sample:
            The last sample in the chain in the exchange format.
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

    def make_internal(self, temper_lpost: bool):
        """
        This function can be called by parallel tempering samplers during
        initialization to allow the inner samplers to adjust to them
        being used as inner samplers. Default: Do nothing.

        Parameters
        ----------
        temper_lpost:
            Whether to temperate the posterior or only the likelihood.
        """
