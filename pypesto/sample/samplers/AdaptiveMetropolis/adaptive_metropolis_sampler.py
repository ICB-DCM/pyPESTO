import numpy as np
from typing import Dict

from ...sampler_v1 import SamplerState, Sampler
from . import adaptive_metropolis_sampler_methods
#from adaptive_metropolis_sampler_parent import SamplerState, Sampler
#import adaptive_metropolis_sampler_methods

# some of the "required" parameters could be given reasonable defaults
# should this be re-written to produce n_samples unique samples?
# at the moment, if a sample is rejected, then there will be duplicate samples
class AdaptiveMetropolisSampler(Sampler):
    '''
    Generate samples of parameter values using the adaptive
    Metropolis-Hastings algorithm.

    Example Usage
    -------------
    ```python
    sampler = AdaptiveMetropolisSampler(settings=settings)
    sampler.sample(100) # burn in
    sampler.save_state(filename)
    first_set_of_samples = sampler.sample(1000) # will be 1100 samples
    sampler.load_state(filename)
    second_set_of_samples = sampler.sample(1000) # again, 1100 samples
    ```

    Restore from a Failed Sampling
    ------------------------------
    A sampler that ends prematurely can be resumed from the failed sampling,
    with the `load_state` method. This example continues from the example
    above. Note that the state should have been saved with the `save_state`
    method. States can be automatically saved periodically with the optional
    `save_period` setting.
    ```python
    restored_sampler = AdaptiveMetropolisSampler()
    restored_sampler.load_state(filename)
    restored_sampler.sample()
    ```

    Required Settings
    -----------------
    The following settings must be provided, as this class provides no
    default values for them.

    log_posterior_callable:
        A function that takes a sample as its only argument, and returns the
        log posterior for that sample. For example, it might run a simulation
        with the sample, then calculate the log objective function with the
        simulation results and measurements.

    sample: Sequence[float]
        Initial parameters sample.

    covariance: np.ndarray
        Estimate of the covariance matrix of the initial parameters sample.

    lower_bounds (respectively, upper_bounds): Sequence[float]
        The lower (respectively, upper) bounds of the parameters in the sample.

    decay_constant: float
        Adaption decay, in (0, 1). Higher values result in faster decays, such
        that later sampling influences the adaption more weakly.

    threshold_sample: int
        Number of samples before adaption decreases significantly.
        Alternatively: a higher value reduces strong early adaption.

    regularization_factor: float
        Factor used to regularize the estimated covariance matrix. Larger
        values result in stronger regularization.

    Optional Settings
    -----------------
    The default values of the following optional parameters are in the
    defaults() method.

    n_samples: int
        Requested number of samples.

    debug: bool
        Return additional information if True.

    save_period: int
        The periodicity that the state shoud be saved with. For example, a
        value of 40 would result in a `save_state` method call every 40
        samples.

    save_filename: string
        The name of the file that the sampler state will be saved to.

    Results
    -------
    From the example usage, the following results could be accessed as keys of
    the `sampler.chain` dictionary. For example, sampler.chain['samples']
    stores all generated samples.

    samples:
        All samples.

    log_posterior:
        log posterior of all samples.

    (if debug) cumulative_chain_acceptance_rate:
        The percentage of samples that were accepted, at each sampling.

    (if debug) covariance_scaling_factor:
        Scaling factor of the estimated covariance matrix, such that there is
        an overall acceptance rate of 23.4%, for each sample.

    (if debug) historical_covariance:
        Estimated covariance matrices of samples from all previous samples,
        at each sampling.
    '''

    def defaults(self, overrides: Dict = {}) -> Dict:
        '''
        The default values of the optional parameters of the sampler, that are
        described in the class docstring, are defined here.

        Returns
        -------
        A dictionary of parameters
        '''
        settings = {
            'n_samples': 100,
            'debug': False,
            'save_period': -1, # -1 for no checkpoints
            'save_filename': 'adaptive_metropolis_saved_state.pickle'
        }

        settings.update(overrides)

        return settings

    def new_chain(self) -> Dict:
        '''
        Initializes a chain, as described in the Sampler class. Results that
        are stored in the chain are described in the Results section of the
        docstring for this class.
        '''
        chain = {
                'samples': np.full([len(self.state.sample), self.state.n_samples], np.nan),
                'samples_log_posterior': np.full([self.state.n_samples], np.nan)
        }

        if self.state.debug:
            chain.update({
                    'cumulative_chain_acceptance_rate': np.full([self.state.n_samples], np.nan),
                    'covariance_scaling_factor': np.full([self.state.n_samples], np.nan),
                    'historical_covariance': np.full([*self.state.covariance.shape, self.state.n_samples], np.nan)
            })

        return chain

    def extend_chain(self, n_samples: int):
        '''
        Extends the current chain, as described in the Sampler class.
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

        if self.state.debug:
            chain['cumulative_chain_acceptance_rate'] = np.concatenate([
                    chain['cumulative_chain_acceptance_rate'],
                    np.full([n_samples], np.nan)
            ])
            chain['covariance_scaling_factor'] = np.concatenate([
                    chain['covariance_scaling_factor'],
                    np.full([n_samples], np.nan)
            ])
            chain['historical_covariance'] = np.concatenate([
                    chain['historical_covariance'],
                    np.full([*self.state.covariance.shape, n_samples], np.nan)
                    ],
                    axis=len(self.state.covariance.shape)
            )

        self.state.n_samples += n_samples
        self.state.chain = chain

    # should accepted_count start at 1 as, in
    # `chain['cumulative_chain_acceptance_rate'][i] = 100*n_accepted/(i+1)`,
    # 1 is added to i to avoid division by zero? Seems like it's better at = 0,
    # then `chain['cumulative_chain_acceptance_rate'] is an actual percentage
    def new_state(self):
        '''
        Creates a new state for entry into the sampling method, `sample`.
        Alternatively: initializes the sampling method `sample()`.
        The required settings, which are described in the Required Settings
        section of the class docstring, must be set prior to calling this
        method.
        '''
        state = SamplerState({k: self.settings[k] for k in self.settings})
        state.n_parameters = len(state.sample)

        state.n_accepted = 0
        state.covariance_scaling_factor = 1.0

        state.historical_mean = state.sample
        state.historical_covariance = state.covariance
        state.covariance = adaptive_metropolis_sampler_methods.regularize_covariance(
                state.covariance,
                state.regularization_factor,
                state.n_parameters,
                MAGIC_DIVIDING_NUMBER = 1000
        )

        state.sample_log_posterior = state.log_posterior_callable(state.sample)

        state.n_sample = 0

        self.state = state
        self.state.settings = self.settings

    def additional_samples(self, n_samples):
        '''
        Determines whether to extend the size of the chain to accommodate
        additional samples.
        Produces samples and stores sampling results in the chain.
        Determines whether the existing chain of a 

        Arguments
        ---------
        n_samples:
            The number of samples to generate.
        '''
        if self.state.n_sample == self.state.n_samples:
            if n_samples == 0:
                # No samples requested. Assume the n_samples setting.
                self.extend_chain(n_samples=self.state.settings['n_samples'])
            # if additional sampling is requested, increased the size of the chains arrays etc.
            elif n_samples > 0:
                self.extend_chain(n_samples=n_samples)
        elif self.state.n_sample < self.state.n_samples and n_samples > 0:
            # Samples requested, but the previous request has not ended.
            # May occur if the previous request failed.
            # May occur if n_samples is specified in settings->__init__(), and
            # then sample(n_samples) is called before sample().
            # What should default behaviour here be? Could extend n_samples
            # by the difference...
            if n_samples + self.state.n_sample > self.state.n_samples:
                raise IndexError('The current requested sampling has not\n'
                                 'yet completed. Resume sampling with\n'
                                 'AdaptiveMetropolisSampler.sample()\n'
                                 'before requesting addition samples.')

    @staticmethod
    def sampling_loop(state: SamplerState, index: int, beta: float) -> SamplerState:
        sampling_result = adaptive_metropolis_sampler_methods.try_sampling(
            state.log_posterior_callable,
            state.sample,
            state.sample_log_posterior,
            state.covariance,
            state.lower_bounds,
            state.upper_bounds,
            state.debug
        )
        state.sample = sampling_result['sample']
        state.sample_log_posterior = sampling_result['log_posterior']
        if sampling_result['accepted']:
            state.n_accepted += 1

        covariance_result = adaptive_metropolis_sampler_methods.estimate_covariance(
            state.historical_mean,
            state.historical_covariance,
            state.sample,
            state.threshold_sample,
            state.decay_constant,
            state.covariance_scaling_factor,
            sampling_result['log_acceptance'],
            state.regularization_factor,
            state.n_parameters,
            index
        )

        state.historical_mean = covariance_result['historical_mean']
        state.historical_covariance = (
            covariance_result['historical_covariance'])
        state.covariance_scaling_factor = (
            covariance_result['covariance_scaling_factor'])
        state.covariance = covariance_result['covariance']

        state.chain['samples'][:,index] = state.sample
        state.chain['samples_log_posterior'][index] = state.sample_log_posterior
        if state.debug:
            # magic number 1 here to avoid division by zero. not an issue in
            # Matlab, as i > 0 there
            state.chain['cumulative_chain_acceptance_rate'][index] = (
                100*state.n_accepted/(index+1))
            state.chain['covariance_scaling_factor'][index] = state.covariance_scaling_factor
            state.chain['historical_covariance'][..., index] = state.historical_covariance

        return state

    def sample(self, n_samples: int = 0, beta: float = 1):
        '''
        Produces samples and stores sampling results in the chain.

        Arguments
        ---------
        n_samples:
            The number of samples to generate.
        beta:
            Value from parallel tempering.

        Returns
        -------
        The chain of results, as described in the Results section of the
        docstring of this class.
        '''
        #print(n_samples)
        self.additional_samples(n_samples)
        state = self.state
        state.chain = self.get_chain()

        #n_samples = n_samples if n_samples > 0 else state.n_samples
        #print((state.n_sample, n_samples))
        for index in range(state.n_sample, self.state.n_samples):
            state = AdaptiveMetropolisSampler.sampling_loop(state, index, beta)

            #self.last

            if state.save_period > 0 and i % state.save_period == 0:
                self.state = state
                self.state.n_sample = index
                self.save_state()

        self.state = state
        self.state.n_sample = self.state.n_samples
        #return [self.state.chain]
        return super().result_as_chains(self.state.chain)

    def get_state(self, key: str = None):
        '''
        An interface to access state values.

        Arguments
        ---------
        key:
            The requested state attribute.

        Returns
        -------
        The state attribute described by the key. If no key is specified, then
        the entire state is returned.
        '''
        if key is None:
            return self.state
        else:
            return self.state.__getattribute__(key)
