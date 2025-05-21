"""Functions for variational inference accessible to the user. Currently only pymc is supported."""

import logging
from time import process_time
from typing import Callable, Optional, Union

import numpy as np

from ..problem import Problem
from ..result import Result
from ..sample.util import bound_n_samples_from_env
from ..store import autosave
from .pymc import PymcVariational

logger = logging.getLogger(__name__)


def variational_fit(
    problem: Problem,
    n_iterations: int,
    method: str = "advi",
    n_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
    start_sigma: Optional[dict[str, np.ndarray]] = None,
    x0: Union[np.ndarray, list[np.ndarray]] = None,
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
                result.optimize_result.list[0]["x"]
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
    if n_samples is None or n_samples == 0:
        # constructing a McmcPtResult object with nearly empty trace_x
        n_samples = 1

    result.sample_result = variational.sample(n_samples)
    result.sample_result.time = t_elapsed

    autosave(
        filename=filename,
        result=result,
        store_type="sample",
        overwrite=overwrite,
    )

    # make pymc object available in result
    # TODO: if needed, we can add a result object for variational inference methods
    result.variational_result = variational
    (
        result.sample_result.variational_parameters_names,
        result.sample_result.variational_parameters,
    ) = variational.get_variational_parameters()
    if filename is not None:
        logger.warning(
            "Variational parameters are not saved in the hdf5 file. You have to save them manually."
        )

    return result
