import logging
from time import process_time
from typing import List, Optional, Union

import numpy as np
from scipy import stats

from ..problem import Problem
from ..result import Result

# from ..store import autosave
from .pymc import PymcSampler
from .util import bound_n_samples_from_env

logger = logging.getLogger(__name__)


def variational_fit(
    problem: Problem,
    n_iterations: int,
    method: str = 'advi',
    # n_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
    start_sigma: Optional[dict[str, np.ndarray]] = None,
    x0: Union[np.ndarray, List[np.ndarray]] = None,
    result: Result = None,
    # filename: Union[str, Callable, None] = None,
    # overwrite: bool = False,
    **kwargs,
) -> Result:
    """
    Call to do parameter sampling.

    Parameters
    ----------
    problem:
        The problem to be solved. If None is provided, a
        :class:`pypesto.AdaptiveMetropolisSampler` is used.
    n_iterations:
        Number of iterations for the optimization.
    method: str or :class:`Inference`
        string name is case-insensitive in:
            -   'advi'  for ADVI
            -   'fullrank_advi'  for FullRankADVI
            -   'svgd'  for Stein Variational Gradient Descent
            -   'asvgd'  for Amortized Stein Variational Gradient Descent
    n_samples:
        Number of samples to generate after optimization.
    random_seed: int
        random seed for reproducibility
    start_sigma: `dict[str, np.ndarray]`
        starting standard deviation for inference, only available for method 'advi'
    x0:
        Initial parameter for the variational optimization. If None, the best parameter
        found in optimization is used.
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
    if n_iterations is not None:
        n_iterations = bound_n_samples_from_env(n_iterations)

    # try to find initial parameters
    if x0 is None:
        result.optimize_result.sort()
        if len(result.optimize_result.list) > 0:
            x0 = problem.get_reduced_vector(
                result.optimize_result.list[0]['x']
            )
        # TODO multiple x0 for PT, #269

    # set variational inference
    variational_approx = PymcSampler()

    # initialize sampler to problem
    variational_approx.initialize(problem=problem, x0=x0)

    # perform the sampling and track time
    t_start = process_time()
    variational_approx.fit(
        n_iterations=n_iterations,
        method=method,
        random_seed=random_seed,
        start_sigma=start_sigma,
        **kwargs,
    )
    t_elapsed = process_time() - t_start
    logger.info("Elapsed time: " + str(t_elapsed))

    # extract results
    # todo: build variational result object
    return variational_approx.data

    # if n_samples is not None:
    #     vi_result.samples = variational_approx.data.sample(n_samples)
    #
    # # record time
    # vi_result.time = t_elapsed
    #
    # # record results
    # result.vi_result = vi_result
    #
    # autosave(
    #     filename=filename,
    #     result=result,
    #     store_type="variational",
    #     overwrite=overwrite,
    # )
    #
    # return result


def eval_variational_log_density(
    x_points: np.ndarray, vi_approx
) -> np.ndarray:
    """
    Evaluate the log density of the variational approximation at x_points.

    Parameters
    ----------
    x_points:
        The points at which to evaluate the log density.
    vi_approx:
        The variational approximation object from PyMC.
    """
    if x_points.ndim == 1:
        x_points = x_points.reshape(1, -1)
    log_density_at_points = np.zeros_like(x_points)
    for i, point in enumerate(x_points):
        log_density_at_points[i] = stats.multivariate_normal.logpdf(
            point, mean=vi_approx.mean.eval(), cov=vi_approx.cov.eval()
        )
    vi_log_density = np.sum(log_density_at_points, axis=-1)
    return vi_log_density
