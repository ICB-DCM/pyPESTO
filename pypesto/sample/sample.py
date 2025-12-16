import logging
from time import process_time
from typing import Callable, Optional, Union

import numpy as np

from ..problem import Problem
from ..result import Result
from ..startpoint import PriorStartpoints
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
    warm_start: float = 1.0,
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
    warm_start:
        Whether to warm start from previous optimization results stored in
        `result` or to sample from the prior. The value is the convex
        combination between the two, with 1.0 meaning only warm start and
        0.0 meaning only sample from prior. If `x0` is provided,
        `warm_start` is ignored.
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

    if warm_start > 1.0 or warm_start < 0.0:
        raise ValueError("warm_start must be in [0, 1].")

    # try to find initial parameters
    if x0 is None:
        if warm_start == 0:
            logger.info("Sampling initial points from prior.")
            get_start_params = PriorStartpoints(check_fval=True)
            x0 = get_start_params.sample(
                n_starts=1,
                lb=problem.lb,
                ub=problem.ub,
                priors=problem.x_priors,
            )[0]
        elif result.optimize_result is not None:
            result.optimize_result.sort()
            if len(result.optimize_result.list) > 0:
                x0 = problem.get_reduced_vector(
                    result.optimize_result.list[0]["x"]
                )
            if warm_start < 1.0 and x0 is not None:
                logger.info(
                    f"Initializing sampling with a warm start from optimization "
                    f"and prior sampling with weight: {warm_start}."
                )
                get_start_params = PriorStartpoints(check_fval=True)
                x0_prior = get_start_params.sample(
                    n_starts=1,
                    lb=problem.lb,
                    ub=problem.ub,
                    priors=problem.x_priors,
                )[0]
                x0 = warm_start * x0 + (1 - warm_start) * x0_prior
        else:
            logger.info(
                "No initial point provided and no optimization result found. Set warm_start to 0 "
                "if you need initial points for sampling."
            )

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
