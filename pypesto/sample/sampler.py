import abc

from types import SimpleNamespace
from typing import Dict, Tuple

import dill

import copy

class SamplerState(object):
    '''
    A generic class that allows instance attributes to be specified with
    initialization, as a dictionary.
    '''
    def __init__(self, attributes: Dict):
        '''
        Arguments
        ---------
        attributes:
            A dictionary that will be converted to attributes of an instance
            of this class.

        Other
        -----
        A default filename is specified in the 'save_filename' attribute of
        instances of this class (remove?).
        '''
        for name, value in attributes.items():
            self.__setattr__(name, value)

# Sampler().state is currently undefined (needs to be defined in child classes)
class Sampler():
    '''
    Generic class that implements save and load functionality. Child samplers
    should implement a `state: SamplerState` attribute that contains the
    current state of the sampler, which can be saved and later restored to
    continue a sampling.

    Options passed to the initializer are used to create a sampler state,
    which can then be used to produce samples, and saved and loaded.
    '''
    def __init__(self, settings: Dict = {}, new_state: bool = True):
        '''
        Arguments
        ---------
        settings:
            Dictionary of settings for the sampler. See specific samplers for
            valid settings. These values will override any corresponding
            default sampler settings.

        new_state:
            Sets up an initial state for the sampler, to begin sampling,
            if True.
        '''
        self.settings = self.defaults(overrides=settings)
        if new_state:
            self.new_state()

    @abc.abstractmethod
    def defaults(self, overrides: Dict = {}) -> Dict:
        '''
        An example of combining default settings with custom settings that were
        specified in the initializer, such that custom settings take precedence
        over default settings.

        Arguments
        ---------
        overrides:
            Specific settings of a child class instance. For keys that exist in
            both the `overrides` and defaults dictionaries, the returned
            dictionary will have the value in the `overrides` dictionary.

        Returns
        -------
        A dictionary of settings. The example here has default values for
        `save_filename` and `n_samples`.
        '''
        return {
            'save_filename': 'sampler_saved_state.pickle',
            'n_samples': '100'
        }.update(overrides)

    @abc.abstractmethod
    def new_state(self) -> None:
        '''
        An example of initializing the state of a child class. Here, state
        attributes corresponding to child class instance settings (that were
        specified, for example, in `__init__`).

        These settings can then be accessed as attributes of self.state
        For example, given the defaults() above, self.state.save_filename
        will be 'sampler_saved_state.pickle'.
        '''
        state = SamplerState({k: self.settings[k] for k in self.settings})
        self.state = state

    @abc.abstractmethod
    def new_chain(self) -> None:
        '''
        Initialize and return a dictionary that chain samples can be stored
        in. The example here assumes that the `sampler.state` `sample` and
        `n_samples` attributes are set.

        Returns
        -------
        A dictionary of initialized chain results, with keys that match the
        results dictionary returned by the `sample` method. The example here
        initializes numpy arrays for 100 samples.
        '''
        chain = {
                'samples': np.full(
                        [len(self.state.sample), self.state.n_samples],
                        np.nan),
                'samples_log_posterior': np.full(
                        [self.state.n_samples], np.nan)
        }
        return chain

    @abc.abstractmethod
    def extend_chain(self, n_samples: int) -> None:
        '''
        Extend the existing chain of the sampler to allow for
        additional sampling.

        Arguments
        ---------
        n_samples:
            The chain will be extended, such that `n_samples`
            addition samples may be stored.
        '''
        chain = self.state.chain
        chain['samples'] = np.apply_along_axis(
                lambda x: np.concatenate([x,
                    np.full([n_samples], np.nan)]),
                axis=1,
                arr=chain['samples'])

        chain['samples_log_posterior'] = np.concatenate([
                chain['samples_log_posterior'],
                np.full([n_samples], np.nan)
        ])

        self.state.n_samples += n_samples
        self.state.chain = chain

    def get_chain(self, ) -> Dict:
        '''
        Returns the chain of the state. If no chain exists, they are created
        with the `new_chain` method. Valid for single-chain samplers.
        '''
        if hasattr(self.state, 'chain'):
            return self.state.chain
        else:
            return self.new_chain()

    # Returns Dict for now, may return a SamplerResult instead
    @abc.abstractmethod
    def sample(self, n_samples:int = 0) -> Dict:
        '''
        Produces samples. The example here assumes the `sampler.state`
        `n_samples` and `sample` attributes are set.

        Arguments
        ---------
        n_samples:
            The number of samples to produce. The default value is 0, because
            this may have already been specified during initialization of a
            child class instance, with the `settings` argument of `__init__`.

        Returns
        -------
        A dictionary of chain vectors, with the following keys. Other sampler
        specific keys may also be returned.
            samples:
                A sequence containing samples.
            samples_log_posterior:
                A sequence containing the log posterior for each sample in
                `samples`.
        '''
        chain = self.get_chain()
        # This loop begins at self.state.n_sample, to allow for additional
        # samplings.
        for i in range(self.state.n_sample, self.state.n_samples):
            chain['samples'][:,i] = np.full(
                            [len(self.state.sample)],
                            i)
            chain['samples_log_posterior'][i] = i

            # Save the state periodically (assumes the existence of a
            # `save_period` state attribute).
            if state.save_period > 0 and i % state.save_period == 0:
                self.state = state
                self.state.chain = chain
                self.state.n_sample = i
                self.save_state()

        # Optional, save the chain in the state, as a "burn-in".
        self.state.chain = chain
        self.state.n_sample = self.state.n_samples
        return chain

    @staticmethod
    def result_as_chains(chain: Dict) -> Dict:
        '''
        Converts the results of single-chain samplers to the same format
        as multiple-chain samplers.

        Valid for single-chain samplers.

        Arguments
        ---------
        chain:
            The result of a single-chain sampler.

        Returns
        -------
        The single chain in the same format as multiple chains.
        '''
        chains = {key: [chain[key]] for key in chain}
        return chains

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

    def save_state(self, filename: str = None):
        '''
        Saves the state of a sampler, such that sampling can be resumed later.
        One use case (in conjunction with the `load_state` and `sample`
        methods) is saving the state of a sampler after burn-in, and then
        repeatedly loading the saved state and obtaining samples.

        Arguments
        ---------
        filename:
            The name of the file that the sampler state will be saved to.
        '''
        if filename is None:
            filename = self.state.save_filename
        with open(filename, 'wb') as f:
            dill.dump(self.state, f)

    def load_state(self, filename: str = None):
        '''
        Loads the state of a sampler, from a file that has been created using
        the `save_state` method.
        '''
        if filename is None:
            filename = self.state.save_filename
        with open(filename, 'rb') as f:
            self.state = dill.load(f)
