import logging
from collections.abc import Iterable

import numpy as np
from scipy.stats import multivariate_normal

from ..problem import Problem
from ..result import ProfilerResult, Result
from .util import initialize_profile

logger = logging.getLogger(__name__)


def approximate_parameter_profile(
    problem: Problem,
    result: Result,
    profile_index: Iterable[int] = None,
    profile_list: int = None,
    result_index: int = 0,
    n_steps: int = 100,
) -> Result:
    """
    Calculate profile approximation.

    Based on an approximation via a normal likelihood centered at the chosen
    optimal parameter value, with the covariance matrix being the Hessian or
    FIM.

    Parameters
    ----------
    problem:
        The problem to be solved.
    result:
        A result object to initialize profiling and to append the profiling
        results to. For example, one might append more profiling runs to a
        previous profile, in order to merge these.
        The existence of an optimization result is obligatory.
    profile_index:
        List with the profile indices to be computed
        (by default all of the free parameters).
    profile_list:
        Integer which specifies whether a call to the profiler should create
        a new list of profiles (default) or should be added to a specific
        profile list.
    result_index:
        Index from which optimization result profiling should be started
        (default: global optimum, i.e., index = 0).
    n_steps:
        Number of profile steps in each dimension.

    Returns
    -------
    The profile results are filled into `result.profile_result`.
    """
    # Handling defaults
    # profiling indices
    if profile_index is None:
        profile_index = problem.x_free_indices

    # create the profile result object (retrieve global optimum) or append to
    # existing list of profiles
    global_opt = initialize_profile(
        problem, result, result_index, profile_index, profile_list
    )

    # extract optimization result
    optimizer_result = result.optimize_result.list[result_index]
    # extract values of interest
    x = optimizer_result.x
    fval = optimizer_result.fval
    hess = problem.get_reduced_matrix(optimizer_result.hess)

    # ratio scaling factor
    ratio_scaling = np.exp(global_opt - fval)

    # we need the hessian - compute if not provided or fishy
    if hess is None or np.isnan(hess).any():
        logger.info("Computing Hessian/FIM as not available in result.")
        hess = problem.objective(
            problem.get_reduced_vector(x), sensi_orders=(2,)
        )

    # inverse of the hessian
    sigma = np.linalg.inv(hess)

    # the steps
    xs = np.linspace(problem.lb_full, problem.ub_full, n_steps).T

    # loop over parameters for profiling
    for i_par in profile_index:
        # not requested or fixed -> compute no profile
        if i_par in problem.x_fixed_indices:
            continue

        i_free_par = problem.full_index_to_free_index(i_par)

        ys = multivariate_normal.pdf(
            xs[i_par], mean=x[i_par], cov=sigma[i_free_par, i_free_par]
        )

        fvals = -np.log(ys)
        ratios = ys / ys.max() * ratio_scaling

        profiler_result = ProfilerResult(
            x_path=xs,
            fval_path=fvals,
            ratio_path=ratios,
        )

        result.profile_result.set_profiler_result(
            profiler_result=profiler_result,
            i_par=i_par,
            profile_list=profile_list,
        )

    return result
