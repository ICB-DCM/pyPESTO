import abc
import numpy as np
from typing import Any, Dict, Tuple, Union

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

    def __init__(self, options: Dict = {}):
        '''
        Arguments
        ---------
        options:
            A dictionary of settings for the sampler. Valid options are
            described in the `default_options` method of specific samplers.
        '''
        self.options = self.translate_options(options)
        #self.options = self.defaults(overrides=settings)
        #if new_state:
        #    self.new_state()

    @staticmethod
    @abc.abstractmethod
    def default_options() -> Dict:
        return {
            'n_samples': 1000
        }

    def translate_options(self, options: Dict = {}) -> Dict:
        '''
        An example of combining default settings with custom settings that were
        specified in the initializer, such that custom settings take precedence
        over default settings.

        Arguments
        ---------
        cls:
            The specific sampler class. This should be passed automatically,
            similar to `self`.

        options:
            Specific settings of a child class instance. For keys that exist in
            both the `options` and default options dictionaries, the returned
            dictionary will have the value in the `options` dictionary.

        Returns
        -------
        A dictionary of options.
        '''
        default_options = self.default_options()
        unknown_keys = set(options.keys()).difference(set(default_options.keys()))
        if unknown_keys:
            raise KeyError(f'Sampler {self.__class__.__name__} cannot handle the following keys'
                           f'{unknown_keys}')
        default_options.update(options) # no longer default options...
        return default_options


    def get_last_sample(self, key: str = None):
        '''
        Returns the last sample (currently valid for single-chain samplers).
        Requires the structure of the chain to be such that the last dimension
        is n_samples (number of iterations).

        Valid for single-chain samplers.

        Arguments
        ---------
        key:
            If specified, the last value for this key in the chain will be
        returned. For example, if key='samples', then the last sample in the
        chain will be returned.

        Returns
        -------
        A dictionary where the keys are the corresponding keys in the
        SamplerState chain, and the values are the last values for the
        respective keys in the SamplerState chain.
        '''
        if key is not None:
            return self.state.chain[key][...,self.state.n_sample - 1]
        else:
            return {
                k: self.state.chain[k][...,self.state.n_sample - 1]
                for k in self.state.chain
            }

    def set_last_sample(self, sample):
        '''
        Sets the most recently generated sample in a SamplerState chain to the
        argument `sample`.

        Valid for single-chain samplers.

        Arguments
        ---------
        sample:
            The sample that will replace the last sample in the SamplerState
        chain.
        '''
        for key in sample:
            self.state.chain[key][...,self.state.n_sample - 1] = sample[key]

    @staticmethod
    def swap_last_samples(
            sampler1: 'Sampler',
            sampler2: 'Sampler'
    ) -> Tuple['Sampler', 'Sampler']:
        '''
        Swaps the last sample (including metadata such as log posterior values)
        of two samplers. Currently valid for single-chain samplers (could be
        extended by instead specifying the chain number(s) for multiple-chain
        samplers).

        Arguments
        ---------
        sampler1:
            An instance of a single-chain Sampler child class.

        sampler2:
            An instance of a single-chain Sampler child class.

        Returns
        -------
        The samplers after their last samples have been swapped.
        '''
        # Not sure if deepcopy is necessary...
        sampler1_last_sample = copy.deepcopy(sampler1.get_last_sample())
        sampler2_last_sample = copy.deepcopy(sampler2.get_last_sample())
        #sampler1.set_last_sample(sampler2.get_last_sample())
        sampler1.set_last_sample(sampler2_last_sample)
        sampler2.set_last_sample(sampler1_last_sample)
        return (sampler1, sampler2)
