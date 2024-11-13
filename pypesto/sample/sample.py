import logging
from time import process_time
from typing import Callable, Optional, Union

import numpy as np

from ..problem import Problem
from ..result import Result
from ..store import autosave
from .adaptive_metropolis import AdaptiveMetropolisSampler
from .sampler import Sampler
from .util import bound_n_samples_from_env

logger = logging.getLogger(__name__)


def sample(
    problem: Problem,
    n_samples: Optional[int],
    sampler: Sampler = None,
    x0: Union[np.ndarray, list[np.ndarray]] = None,
    result: Result = None,
    filename: Union[str, Callable, None] = None,
    overwrite: bool = False,
) -> Result:
    """
    Call to do parameter sampling.

    Parameters
    ----------
    problem:
        The problem to be solved. If None is provided, a
        :class:`pypesto.AdaptiveMetropolisSampler` is used.
    n_samples:
        Number of samples to generate. `None` can be used if the sampler does
        not use `n_samples`.
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
    filename:
        Name of the hdf5 file, where the result will be saved. Default is
        None, which deactivates automatic saving. If set to
        "Auto" it will automatically generate a file named
        `year_month_day_profiling_result.hdf5`.
        Optionally a method, see docs for `pypesto.store.auto.autosave`.
    overwrite:
        Whether to overwrite `result/sampling` in the autosave file
        if it already exists.

    Returns
    -------
    result:
        A result with filled in sample_options part.
    """
    # prepare result object
    if result is None:
        result = Result(problem)

    # number of samples
    if n_samples is not None:
        n_samples = bound_n_samples_from_env(n_samples)

    # try to find initial parameters
    if x0 is None:
        result.optimize_result.sort()
        if len(result.optimize_result.list) > 0:
            x0 = problem.get_reduced_vector(
                result.optimize_result.list[0]["x"]
            )
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
    logger.info("Elapsed time: " + str(t_elapsed))

    # extract results
    sampler_result = sampler.get_samples()

    # record time
    sampler_result.time = t_elapsed

    # record results
    result.sample_result = sampler_result

    autosave(
        filename=filename,
        result=result,
        store_type="sample",
        overwrite=overwrite,
    )

    return result
