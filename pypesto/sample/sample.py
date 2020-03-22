import logging

try:
    import pymc3
    from pymc3.step_methods import (NUTS, HamiltonianMC, Metropolis, Slice)
    from pymc3.distributions import Uniform
    import theano
except ImportError:
    pass

from ..problem import Problem
from ..result import Result
from .result import SamplerResult
from .sampler import Sampler

logger = logging.getLogger(__name__)


def parameter_sample(
        problem: Problem,
        sampler: Sampler,
        draws: int = 1000,
        result: Result = None
) -> Result:
    """
    This is the main function to call to do parameter sampling.

    Parameters
    ----------
    problem:
        The problem to be solved.
    sampler:
        The sampler to perform the actual sampling.
    draws:
        Number of samples to draw.
    result:
        A result to write to.

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



