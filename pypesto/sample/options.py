from typing import Dict, Union

class SamplerOptions(dict):
    """
    Options for parameter sampling.

    Parameters
    ----------
    n_samples:
        Number of samples to generate (chain length). If multiple chains are
        employed, the total number of samples generated will be larger, usually
        `chains` * `n_samples`.
    n_chains:
        The number of chains to sample. Running independent chains is important
        for some convergence statistics and can also reveal multiple modes in
        the posterior.
    """

    def __init__(self,
                 n_samples: int = 1000,
                 n_chains: int = 1):
        super().__init__()

        self.n_samples = n_samples
        self.n_chains = n_chains

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def create_instance(maybe_options: Union[Dict, 'SamplerOptions']):
        """
        Returns a valid options object.

        Parameters
        ----------
        maybe_options: OptimizeOptions or dict
        """
        if isinstance(maybe_options, SamplerOptions):
            return maybe_options
        options = SamplerOptions(**maybe_options)
        return options
