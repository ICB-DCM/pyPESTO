import numpy as np
import scipy.stats
from typing import Any, Dict, Tuple, Iterable

from ..objective.constants import GRAD
from ..problem import Problem
from ..result import Result, ProfileResult
from .result import ProfilerResult


def chi2_quantile_to_ratio(alpha: float = 0.95, df: int = 1):
    """
    Transform lower tail probability `alpha` for a chi2 distribution with `df`
    degrees of freedom to a profile likelihood ratio threshold.

    Parameters
    ----------
    alpha:
        Lower tail probability, defaults to 95% interval.
    df:
        Degrees of freedom. Defaults to 1.

    Returns
    -------
    ratio:
        Corresponds to a likelihood ratio.
    """
    quantile = scipy.stats.chi2.ppf(alpha, df=df)
    ratio = np.exp(-quantile / 2)
    return ratio


def calculate_approximate_ci(
        xs: np.ndarray, ratios: np.ndarray, confidence_ratio: float
) -> Tuple[float, float]:
    """
    Calculate approximate confidence interval based on profile. Interval
    bounds are linerly interpolated.

    Parameters
    ----------
    xs:
        The ordered parameter values along the profile for the coordinate of
        interest.
    ratios:
        The likelihood ratios corresponding to the parameter values.
    confidence_ratio:
        Minimum confidence ratio to base the confidence interval upon, as
        obtained via `pypesto.profile.chi2_quantile_to_ratio`.

    Returns
    -------
    lb, ub:
        Bounds of the approximate confidence interval.
    """
    # extract indices where the ratio is larger than the minimum ratio
    indices, = np.where(ratios >= confidence_ratio)
    l_ind, u_ind = indices[0], indices[-1]

    # lower bound
    if l_ind == 0:
        lb = xs[l_ind]
    else:
        # linear interpolation with next smaller value
        ind = [l_ind - 1, l_ind]
        lb = np.interp(confidence_ratio, ratios[ind], xs[ind])

    # upper bound
    if u_ind == len(ratios) - 1:
        ub = xs[u_ind]
    else:
        # linear interpolation with next larger value
        ind = [u_ind + 1, u_ind]  # flipped as interp expects increasing xs
        ub = np.interp(confidence_ratio, ratios[ind], xs[ind])

    return lb, ub


def initialize_profile(
        problem: Problem,
        result: Result,
        result_index: int,
        profile_index: Iterable[int],
        profile_list: int
) -> float:
    """
    This function initializes profiling based on a previous optimization.

    Parameters
    ----------
    problem:
        The problem to be solved.
    result:
        A result object to initialize profiling and to append the profiling
        results to. For example, one might append more profiling runs to a
        previous profile, in order to merge these.
        The existence of an optimization result is obligatory.
    result_index:
        index from which optimization result profiling should be started
    profile_index:
        array with parameter indices, whether a profile should
        be computed (1) or not (0)
        Default is all profiles should be computed
    profile_list:
        integer which specifies whether a call to the profiler should create
        a new list of profiles (default) or should be added to a specific
        profile list

    Returns
    -------
    global_opt:
        log-posterior at global optimum.
    """
    # Check whether an optimization result is existing
    if result.optimize_result is None:
        raise ValueError(
            "Optimization has to be carried out before profiling can be done.")

    tmp_optimize_result = result.optimize_result.as_list()

    # Check if new profile_list is to be created
    if profile_list is None:
        result.profile_result.append_empty_profile_list()

    # get the log-posterior of the global optimum
    global_opt = tmp_optimize_result[0]["fval"]

    # fill the list with optimization results where necessary
    fill_profile_list(
        profile_result=result.profile_result,
        optimizer_result=tmp_optimize_result[result_index],
        profile_index=profile_index,
        profile_list=profile_list,
        problem_dimension=problem.dim_full,
        global_opt=global_opt)

    # return the log-posterior of the global optimum (needed in order to
    # compute the log-posterior-ratio)
    return global_opt


def fill_profile_list(
        profile_result: ProfileResult,
        optimizer_result: Dict[str, Any],
        profile_index: Iterable[int],
        profile_list: int,
        problem_dimension: int,
        global_opt: float
) -> None:
    """
    This is a helper function for initialize_profile

    Parameters
    ----------
    profile_result:
        A list of profiler result objects.
    optimizer_result:
        A local optimization result.
    profile_index:
        array with parameter indices, whether a profile should
        be computed (1) or not (0).
        Default is all profiles should be computed.
    profile_list:
        integer which specifies whether a call to the profiler should
        create a new list of profiles (default) or should be added to a
        specific profile list.
    problem_dimension:
        number of parameters in the unreduced problem.
    global_opt:
        log-posterior at global optimum.
    """

    if optimizer_result[GRAD] is not None:
        gradnorm = np.linalg.norm(optimizer_result[GRAD])
    else:
        gradnorm = None

    # create blank profile
    new_profile = ProfilerResult(
        x_path=optimizer_result["x"],
        fval_path=np.array([optimizer_result["fval"]]),
        ratio_path=np.array([np.exp(global_opt - optimizer_result["fval"])]),
        gradnorm_path=gradnorm,
        exitflag_path=optimizer_result["exitflag"],
        time_path=np.array([0.]),
        time_total=0.,
        n_fval=0,
        n_grad=0,
        n_hess=0,
        message=None)

    if profile_list is None:
        # All profiles have to be created from scratch
        for i_parameter in range(0, problem_dimension):
            if i_parameter in profile_index:
                # Should we create a profile for this index?
                profile_result.append_profiler_result(new_profile)
            else:
                # if no profile should be computed for this parameter
                profile_result.append_profiler_result(None)

    else:
        for i_parameter in range(0, problem_dimension):
            # We append to an existing list
            if i_parameter in profile_index:
                # Do we have to create a new profile?
                create_new = (profile_result.list[profile_list][i_parameter]
                              is None)
                if create_new:
                    profile_result.set_profiler_result(
                        new_profile, i_parameter)
