import logging

try:
    import pymc3
    from pymc3.step_methods import (NUTS, HamiltonianMC, Metropolis, Slice)
    from pymc3.distributions import Uniform
    import theano
except ImportError:
    pass

from pypesto import Result
from ..optimize import OptimizeOptions
from .sampler import SamplerResult

logger = logging.getLogger(__name__)


class SampleOptions(dict):
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
        the posterior. If None, then set to either cores or 2, whichever is
        larger. For SMC the number of chains is the number of
        draws.

    cores: int
        The number of chains to run in parallel. If None, set to the number of
        CPUs in the system, but at most 4 (for ‘SMC’ defaults to 1). Keep in
        mind that some chains might themselves be multithreaded via openmp or
        BLAS. In those cases it might be faster to set this to 1.

    tune: int
        Number of iterations to tune, defaults to 500. Ignored when using
        ‘SMC’. Samplers adjust the step sizes, scalings or similar during
        tuning. Tuning samples will be drawn in addition to the number
        specified in the draws argument, and will be discarded unless
        discard_tuned_samples is set to False.

    nuts_kwargs: dict
        Options for the NUTS sampler. See the docstring of NUTS for a complete
        list of options. Common options are:

            target_accept: float in [0, 1]. The step size is tuned such that
            we approximate this acceptance rate. Higher values like 0.9 or
            0.95 often work better for problematic posteriors.

            max_treedepth: The maximum depth of the trajectory tree.

            step_scale: float, default 0.25 The initial guess for the step size
            scaled down by 1/n**(1/4). If you want to pass options to other
            step methods, please use step_kwargs.

    step_kwargs: dict
        Options for step methods. Keys are the lower case names of the step
        method, values are dicts of keyword arguments. You can find a full list
        of arguments in the docstring of the step methods. If you want to pass
        arguments only to nuts, you can use nuts_kwargs.

    random_seed: int or list of ints
        A list is accepted if cores is greater than one.

    discard_tuned_samples: bool
        Whether to discard posterior samples of the tune interval.
        Ignored when using ‘SMC’

    compute_convergence_checks: bool, default=True
        Whether to compute sampler statistics like gelman-rubin and
        effective_n. Ignored when using ‘SMC’

    use_mmap: bool, default=False
        Whether to use joblib’s memory mapping to share numpy arrays when
        sampling across multiple cores. Ignored when using ‘SMC’

    """

    def __init__(self,
                 n_init=None,
                 chains=1,
                 cores=1,
                 tune=500,
                 nuts_kwargs=None,
                 step_kwargs=None,
                 random_seed=None,
                 discard_tuned_samples=None,
                 compute_convergence_checks=None,
                 use_mmap=None):
        super().__init__()

        self.n_init = n_init
        self.chains = chains
        self.cores = cores
        self.tune = tune
        self.nuts_kwargs = nuts_kwargs
        self.step_kwargs = step_kwargs
        self.random_seed = random_seed
        self.discard_tuned_samples = discard_tuned_samples
        self.compute_convergence_checks = compute_convergence_checks
        self.use_mmap = use_mmap

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
        if isinstance(maybe_options, SampleOptions):
            return maybe_options
        options = SampleOptions(**maybe_options)
        return options


def parameter_sample(
        problem,
        result,
        result_index=0,
        draws=1000,
        sample_options=None) -> Result:
    """
    This is the main function to call to do parameter sampling.

    Parameters
    ----------

    problem: pypesto.Problem
        The problem to be solved.

    result: pypesto.Result
        A result object to initialize profiling and to append the profiling
        results to. For example, one might append more profiling runs to a
        previous profile, in order to merge these.
        The existence of an optimization result is obligatory.

    draws: int, optional
        number of samples to draw

    sample_options: pypesto.SampleOptions, optional
        Various options applied to the sampling.

    """

    # check optimization options
    if sample_options is None:
        sample_options = SampleOptions()
    sample_options = SampleOptions.assert_instance(sample_options)

    # create the sample result object
    result.sample_result = SamplerResult([])

    pymc3_kwargs = {
        option: value
        for option, value in SampleOptions.items()
        if value is not None
    }

    if not hasattr(problem.objective,'max_sensi_order') or \
            problem.objective.max_sensi_order > 0:
        step_method = (NUTS, Metropolis, HamiltonianMC, Slice)
    else:
        step_method = (Metropolis, Slice)

    pymc3.sampling.sample(
        draws=draws,
        init='jitter+adapt_diag',
        start=dict(),  # TODO: extract from optimization
        step_method=step_method,
        progressbar=False,
        live_plot=False,
        **pymc3_kwargs,
    )



    # return
    return result


class PyMC3Model(pymc3.model.Model):

    def __init__(self, problem):

        super(PyMC3Model, self).__init__('', None)


        for ix, name in enumerate(problem.x_names):
            self.Var(name, Uniform.dist(
                lower=problem.lb[ix],
                upper=problem.ub[ix]
            ))

        posterior = PosteriorTheano(problem.objective)

        pymc3.Deterministic('model_posterior', posterior(theta))


class PosteriorTheano(theano.Op):
    """
    Theano implementation of the posterior
    """
    itypes = None
    otypes = None


    def __init__(self, objective):
        self.objective = objective
        itypes = [theano.dscalar]
        otypes = [theano.dscalar]

    def perform(self, node, inputs, outputs):
        theta, = inputs
        fval = self.objective._call_mode_fun(theta, (0,))
        outputs[0][0] = tt.exp(fval)

    def grad(self, inputs, g):
        theta, = inputs
        sllh = self.objective._call_mode_fun(theta, (1,))
        return sllh
