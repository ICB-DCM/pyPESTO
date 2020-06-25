import logging
import numpy as np
from typing import List, Union
from time import process_time

from ..problem import Problem
from ..result import Result
from .sampler import Sampler
from .adaptive_metropolis import AdaptiveMetropolisSampler

logger = logging.getLogger(__name__)


def sample(
        problem: Problem,
        n_samples: int,
        sampler: Sampler = None,
        x0: Union[np.ndarray, List[np.ndarray]] = None,
        result: Result = None
) -> Result:
    """
    This is the main function to call to do parameter sampling.

    Parameters
    ----------
    problem:
        The problem to be solved. If None is provided, a
        :class:`pypesto.AdaptiveMetropolisSampler` is used.
    n_samples:
        Number of samples to generate.
    sampler:
        The sampler to perform the actual sampling.
    x0:
        Initial parameter for the Markov chain. If None, the best parameter
        found in optimization is used. Note that some samplers require an
        initial parameter, some may ignore it. x0 can also be a list,
        to have separate starting points for parallel tempering chains.
    result:
        A result to write to. If None provided, one is created from the
        problem.

    Returns
    -------
    result:
        A result with filled in sample_options part.
    """
    # prepare result object
    if result is None:
        result = Result(problem)

    # try to find initial parameters
    if x0 is None:
        result.optimize_result.sort()
        if len(result.optimize_result.list) > 0:
            x0 = problem.get_reduced_vector(
                result.optimize_result.list[0]['x'])
        # TODO multiple x0 for PT, #269

    # set sampler
    if sampler is None:
        sampler = AdaptiveMetropolisSampler()

    # initialize sampler to problem
    sampler.initialize(problem=problem, x0=x0)

    # perform the sampling and track time
    t_start = process_time()
    sampler.sample(n_samples=n_samples)
    t_elapsed = process_time() - t_start
    logger.info("Elapsed time: "+str(t_elapsed))

    # extract results
    sampler_result = sampler.get_samples()

    # record time
    sampler_result.time = t_elapsed

    # record results
    result.sample_result = sampler_result

    return result
