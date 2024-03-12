import logging
from time import process_time
from typing import Callable, List, Optional, Union

import numpy as np
from scipy import stats

from ..problem import Problem
from ..result import Result
from ..sample.util import bound_n_samples_from_env
from ..store import autosave
from .pymc import PymcVariational

logger = logging.getLogger(__name__)


def variational_fit(
    problem: Problem,
    n_iterations: int,
    method: str = 'advi',
    n_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
    start_sigma: Optional[dict[str, np.ndarray]] = None,
    x0: Union[np.ndarray, List[np.ndarray]] = None,
    result: Result = None,
    filename: Union[str, Callable, None] = None,
    overwrite: bool = False,
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
    method: str or :class:`Inference` of pymc (only interface currently supported)
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

    # set variational inference
    # currently we only support pymc
    variational = PymcVariational()

    # initialize sampler to problem
    variational.initialize(problem=problem, x0=x0)

    # perform the sampling and track time
    t_start = process_time()
    variational.fit(
        n_iterations=n_iterations,
        method=method,
        random_seed=random_seed,
        start_sigma=start_sigma,
        **kwargs,
    )
    t_elapsed = process_time() - t_start
    logger.info("Elapsed time: " + str(t_elapsed))

    # extract results and save samples to pypesto result
    if n_samples is not None and n_samples > 0:
        result.sample_result = variational.sample(n_samples)
        result.sample_result.time = t_elapsed

        autosave(
            filename=filename,
            result=result,
            store_type="sample",
            overwrite=overwrite,
        )

    result.variational_result = variational
    if filename is not None:
        logger.warning(
            'Internal pymc object is not saved. '
            'Please use `save_internal_object` method to save the internal pymc object.'
        )
    return result


def eval_variational_log_density(
    x_points: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """
    Evaluate the log density of the variational approximation at x_points.

    Parameters
    ----------
    x_points:
        The points at which to evaluate the log density.
    mean:
        The mean of the Gaussian variational family.
    cov:
        The cov of the Gaussian variational family.
    """
    if x_points.ndim == 1:
        x_points = x_points.reshape(1, -1)
    log_density_at_points = np.zeros_like(x_points)
    for i, point in enumerate(x_points):
        log_density_at_points[i] = stats.multivariate_normal.logpdf(
            point, mean=mean, cov=cov
        )
    vi_log_density = np.sum(log_density_at_points, axis=-1)
    return vi_log_density
