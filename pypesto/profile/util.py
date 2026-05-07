"""Utility function for profile module."""

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats

from ..C import GRAD
from ..problem import Problem
from ..result import ProfileResult, ProfilerResult, Result
from .options import ProfileOptions

PROFILE_STEP_PRECHECK_NOMINAL_WARN_THRESHOLD = 200
PROFILE_STEP_PRECHECK_DENSE_WARN_THRESHOLD = 1000


@dataclass(frozen=True)
class ResolvedProfileStepSizes:
    """
    Effective step sizes for one profiled parameter.

    Attributes
    ----------
    default_step_size:
        Effective default step size after combining absolute and relative
        settings.
    min_step_size:
        Effective minimum step size after combining absolute and relative
        settings.
    max_step_size:
        Effective maximum step size after combining absolute and relative
        settings.
    span:
        Full parameter span `ub - lb` if a finite positive span was available
        for a `lin`-scale parameter, else `None`.
    uses_relative_min:
        Whether the effective minimum step size is larger than the configured
        absolute minimum due to the relative setting.
    uses_relative_default:
        Whether the effective default step size is larger than the configured
        absolute default due to the relative setting.
    uses_relative_max:
        Whether the effective maximum step size is larger than the configured
        absolute maximum due to the relative setting.
    """

    default_step_size: float
    min_step_size: float
    max_step_size: float
    span: float | None
    uses_relative_min: bool
    uses_relative_default: bool
    uses_relative_max: bool


def chi2_quantile_to_ratio(alpha: float = 0.95, df: int = 1):
    """
    Compute profile likelihood threshold.

    Transform lower tail probability `alpha` for a chi2 distribution with `df`
    degrees of freedom to a profile likelihood ratio threshold.

    Parameters
    ----------
    alpha:
        Lower tail probability, defaults to 95% interval.
    df:
        Degrees of freedom.

    Returns
    -------
    The computed likelihood ratio threshold.
    """
    quantile = scipy.stats.chi2.ppf(alpha, df=df)
    ratio = np.exp(-quantile / 2)
    return ratio


def calculate_approximate_ci(
    xs: np.ndarray, ratios: np.ndarray, confidence_ratio: float
) -> tuple[float, float]:
    """
    Calculate approximate confidence interval based on profile.

    Interval bounds are linearly interpolated.

    Parameters
    ----------
    xs:
        The ordered parameter values along the profile for the coordinate of
        interest.
    ratios:
        The likelihood ratios corresponding to the parameter values.
    confidence_ratio:
        Minimum confidence ratio to base the confidence interval upon, as
        obtained via :func:`pypesto.profile.chi2_quantile_to_ratio`.

    Returns
    -------
    Bounds of the approximate confidence interval.
    """
    # extract indices where the ratio is larger than the minimum ratio
    (indices,) = np.where(ratios >= confidence_ratio)
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


def resolve_profile_step_sizes(
    problem: Problem,
    i_par: int,
    options: ProfileOptions,
) -> ResolvedProfileStepSizes:
    """
    Resolve effective profile step sizes for one parameter.

    The profiling options expose absolute step-size settings for all
    parameters and relative step-size settings for wide `lin`-scale
    parameters. This helper combines both into one set of effective values
    for the profiled parameter.

    For `lin`-scale parameters with finite positive span `ub - lb`, the
    effective step sizes are computed as the maxima of the corresponding
    absolute and relative settings. For `log` and `log10` parameters, or if
    the span is not finite and positive, the absolute settings are used
    unchanged.

    Parameters
    ----------
    problem:
        The parameter estimation problem containing bounds and scales.
    i_par:
        Index of the profiled parameter in full dimension.
    options:
        Profile options containing absolute and relative step-size settings.

    Returns
    -------
    resolved_steps:
        A :class:`ResolvedProfileStepSizes` dataclass containing the effective
        minimum, default, and maximum step sizes for the profiled parameter,
        together with metadata describing whether relative settings were
        active.
    """
    default_step_size = options.default_step_size
    min_step_size = options.min_step_size
    max_step_size = options.max_step_size
    span = None
    uses_relative_min = False
    uses_relative_default = False
    uses_relative_max = False

    scale = str(problem.x_scales[i_par]).lower()
    if scale == "lin":
        candidate_span = float(problem.ub_full[i_par] - problem.lb_full[i_par])
        if np.isfinite(candidate_span) and candidate_span > 0:
            # Compute relative step sizes from the parameter span.
            span = candidate_span
            relative_min = options.min_step_size_relative * span
            relative_default = options.default_step_size_relative * span
            relative_max = options.max_step_size_relative * span

            # Use the larger of the absolute and relative step-size settings.
            min_step_size = max(min_step_size, relative_min)
            default_step_size = max(default_step_size, relative_default)
            max_step_size = max(
                max_step_size,
                relative_max,
                default_step_size,
            )

            # Record whether the relative settings changed the effective ones.
            uses_relative_min = min_step_size > options.min_step_size
            uses_relative_default = (
                default_step_size > options.default_step_size
            )
            uses_relative_max = max_step_size > options.max_step_size

    return ResolvedProfileStepSizes(
        default_step_size=default_step_size,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        span=span,
        uses_relative_min=uses_relative_min,
        uses_relative_default=uses_relative_default,
        uses_relative_max=uses_relative_max,
    )


def precheck_profile_step_size(
    current_profile: ProfilerResult,
    problem: Problem,
    i_par: int,
    par_direction: int,
    options: ProfileOptions,
) -> None:
    """
    Precheck whether the current step-size settings are suspiciously small.

    The check compares the remaining span in the current profiling direction
    against the resolved effective default and minimum step sizes and warns, or
    raises, if the resulting number of expected profile points exceeds
    configured heuristic thresholds. For `log` and `log10` parameters, the
    span and step sizes are interpreted on the transformed optimization scale.

    Parameters
    ----------
    current_profile:
        The current profile path, used to determine the current parameter
        value.
    problem:
        The parameter estimation problem containing bounds and scales.
    i_par:
        Index of the profiled parameter in full dimension.
    par_direction:
        Profiling direction, either `-1` for descending or `1` for ascending.
    options:
        Profile options controlling the precheck behavior and step-size
        settings.
    """
    if options.step_size_precheck_mode == "off":
        return

    scale = str(problem.x_scales[i_par]).lower()
    resolved_steps = resolve_profile_step_sizes(problem, i_par, options)

    x0 = float(current_profile.x_path[i_par, -1])
    if par_direction == -1:
        direction_label = "descending"
        available_span = x0 - float(problem.lb_full[i_par])
    elif par_direction == 1:
        direction_label = "ascending"
        available_span = float(problem.ub_full[i_par]) - x0
    else:
        raise ValueError("par_direction must be either -1 or 1.")

    if not np.isfinite(available_span) or available_span <= 0:
        return

    nominal_count = available_span / resolved_steps.default_step_size
    dense_count = available_span / resolved_steps.min_step_size

    # Check whether the expected number of steps exceeds
    # the configured thresholds and emit a warning if so.
    nominal_warn = nominal_count > PROFILE_STEP_PRECHECK_NOMINAL_WARN_THRESHOLD
    dense_warn = dense_count > PROFILE_STEP_PRECHECK_DENSE_WARN_THRESHOLD
    if not nominal_warn and not dense_warn:
        return

    parameter_name = problem.x_names[i_par]
    message = (
        "Profiling precheck: parameter "
        f"'{parameter_name}' ({scale}, {direction_label}) may require many "
        "profile steps. "
        f"available_span={available_span:.6g}, "
        f"effective_default_step_size={resolved_steps.default_step_size:.6g}, "
        f"effective_min_step_size={resolved_steps.min_step_size:.6g}, "
        f"estimated nominal steps={nominal_count:.1f}, "
        f"estimated worst-case steps={dense_count:.1f}. "
        "Consider increasing the step sizes."
    )
    if not options.whole_path:
        message += (
            " whole_path=False, so this is a bound-based upper estimate and "
            f"the run may stop earlier at ratio_min={options.ratio_min:.6g}."
        )
    if dense_warn:
        message += " Worst-case step count is especially high."

    if dense_warn and options.step_size_precheck_mode == "raise":
        raise ValueError(message)

    warnings.warn(message, UserWarning, stacklevel=2)


def initialize_profile(
    problem: Problem,
    result: Result,
    result_index: int,
    profile_index: Iterable[int],
    profile_list: int,
) -> float:
    """
    Initialize profiling based on a previous optimization.

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
            "Optimization has to be carried out before profiling can be done."
        )

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
        global_opt=global_opt,
    )

    # return the log-posterior of the global optimum (needed in order to
    # compute the log-posterior-ratio)
    return global_opt


def fill_profile_list(
    profile_result: ProfileResult,
    optimizer_result: dict[str, Any],
    profile_index: Iterable[int],
    profile_list: int,
    problem_dimension: int,
    global_opt: float,
) -> None:
    """Fill a ProfileResult.

    Helper function for `initialize_profile`.

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
        gradnorm = np.nan

    # create blank profile
    new_profile = ProfilerResult(
        x_path=optimizer_result["x"][..., np.newaxis],
        fval_path=np.array([optimizer_result["fval"]]),
        ratio_path=np.array([np.exp(global_opt - optimizer_result["fval"])]),
        gradnorm_path=np.array([gradnorm]),
        exitflag_path=np.array([optimizer_result["exitflag"]]),
        time_path=np.array([0.0]),
        color_path=np.array([[1, 0, 0, 1]]),
        time_total=0.0,
        n_fval=0,
        n_grad=0,
        n_hess=0,
        message=None,
    )

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
                create_new = (
                    profile_result.list[profile_list][i_parameter] is None
                )
                if create_new:
                    profile_result.set_profiler_result(
                        new_profile, i_parameter
                    )
