class SamplerOptions(dict):
    """
    Options for parameter sampling.

    Parameters
    ----------
    n_init: int
        Number of iterations of initializer. Only works for ‘nuts’ and ‘ADVI’.
        If ‘ADVI’, number of iterations, if ‘nuts’, number of draws.
    chains: int
        The number of chains to sample. Running independent chains is important
        for some convergence statistics and can also reveal multiple modes in
        the posterior.
    tune: int
        Number of iterations to tune, defaults to 500. Ignored when using
        ‘SMC’. Samplers adjust the step sizes, scalings or similar during
        tuning. Tuning samples will be drawn in addition to the number
        specified in the draws argument, and will be discarded unless
        discard_tuned_samples is set to False.
    """

    def __init__(self):
        super().__init__()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def create_instance(maybe_options):
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
