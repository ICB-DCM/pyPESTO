"""Utility function for profile module."""

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

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

    The minimum, default, and maximum values always come from the same
    step-size family.

    Attributes
    ----------
    mode:
        Selected step-size family, either `"absolute"` or `"relative"`.
    default_step_size:
        Resolved default step size.
    min_step_size:
        Resolved minimum step size.
    max_step_size:
        Resolved maximum step size.
    span:
        Parameter span `ub - lb` on the optimization scale.
    """

    mode: Literal["absolute", "relative"]
    default_step_size: float
    min_step_size: float
    max_step_size: float
    span: float


ResolvedProfileStepSizeMap = dict[int, ResolvedProfileStepSizes]


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


def validate_profile_parameter_bounds(problem: Problem, i_par: int) -> float:
    """Validate finite profile bounds for one parameter and return its span."""
    lb = float(problem.lb_full[i_par])
    ub = float(problem.ub_full[i_par])
    if not np.isfinite(lb) or not np.isfinite(ub):
        raise ValueError(
            "Profiling requires finite lower and upper bounds for parameter "
            f"'{problem.x_names[i_par]}' (index={i_par})."
        )
    span = ub - lb
    if span <= 0:
        raise ValueError(
            "Profiling requires an upper bound greater than the lower bound "
            f"for parameter '{problem.x_names[i_par]}' (index={i_par})."
        )
    return span


def resolve_profile_step_sizes(
    problem: Problem,
    i_par: int,
    options: ProfileOptions,
) -> ResolvedProfileStepSizes:
    """
    Resolve effective profile step sizes for one parameter.

    Relative step sizes are scaled by the parameter span `ub - lb`. If the
    resolved relative default is at least as large as the absolute default,
    the full relative family is used. Otherwise the full absolute family is
    used.

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
        Resolved step sizes and selection metadata.
    """
    # Bounds are required here because relative steps are defined from the
    # finite parameter span on the optimization scale.
    span = validate_profile_parameter_bounds(problem, i_par)

    if options.default_step_size_relative > 0:
        relative_default_step_size = options.default_step_size_relative * span
        relative_min_step_size = options.min_step_size_relative * span
        relative_max_step_size = options.max_step_size_relative * span

        # Select one complete step-size family based on the default step size.
        if (
            options.default_step_size_absolute == 0
            or relative_default_step_size >= options.default_step_size_absolute
        ):
            return ResolvedProfileStepSizes(
                mode="relative",
                default_step_size=relative_default_step_size,
                min_step_size=relative_min_step_size,
                max_step_size=relative_max_step_size,
                span=span,
            )

    return ResolvedProfileStepSizes(
        mode="absolute",
        default_step_size=options.default_step_size_absolute,
        min_step_size=options.min_step_size_absolute,
        max_step_size=options.max_step_size_absolute,
        span=span,
    )


def resolve_profile_step_sizes_for_parameters(
    problem: Problem,
    parameter_indices: Iterable[int],
    options: ProfileOptions,
) -> ResolvedProfileStepSizeMap:
    """Resolve effective profile step sizes for multiple parameters."""
    return {
        i_par: resolve_profile_step_sizes(problem, i_par, options)
        for i_par in parameter_indices
    }


def _format_profile_step_size_resolution_summary(
    problem: Problem,
    i_par: int,
    resolved_steps: ResolvedProfileStepSizes,
) -> str:
    """Create a one-line summary of the resolved step-size family."""
    scale = str(problem.x_scales[i_par]).lower()
    parameter_name = problem.x_names[i_par]

    return (
        "Resolved profile step sizes for "
        f"{parameter_name} (index={i_par}): "
        f"family={resolved_steps.mode}, "
        f"scale={scale}, span={resolved_steps.span}, "
        f"min={resolved_steps.min_step_size}, "
        f"default={resolved_steps.default_step_size}, "
        f"max={resolved_steps.max_step_size}."
    )


def precheck_profile_step_size(
    current_profile: ProfilerResult,
    problem: Problem,
    i_par: int,
    par_direction: int,
    options: ProfileOptions,
    resolved_steps: ResolvedProfileStepSizes,
) -> None:
    """
    Warn or raise if the resolved step sizes imply many profile steps.

    Two estimates are formed: a nominal one from the default step size and a
    worst-case one from the minimum step size. In ``"raise"`` mode, an error
    is raised only when the worst-case estimate is excessive; a merely large
    nominal estimate only triggers a warning, so valid runs are not broken.

    Parameters
    ----------
    current_profile:
        Current profile path.
    problem:
        The parameter estimation problem.
    i_par:
        Index of the profiled parameter in full dimension.
    par_direction:
        Profiling direction, either `-1` for descending or `1` for ascending.
    options:
        Profile options.
    resolved_steps:
        Pre-resolved step sizes for the profiled parameter.
    """
    if options.step_size_precheck_mode == "off":
        return

    # Estimate how much of the bounded parameter range is left in this
    # profiling direction.
    x0 = float(current_profile.x_path[i_par, -1])
    if par_direction == -1:
        available_span = x0 - float(problem.lb_full[i_par])
    elif par_direction == 1:
        available_span = float(problem.ub_full[i_par]) - x0
    else:
        raise ValueError("par_direction must be either -1 or 1.")

    if not np.isfinite(available_span) or available_span <= 0:
        return

    # Use the resolved default and minimum steps as nominal and dense estimates.
    nominal_count = available_span / resolved_steps.default_step_size
    dense_count = available_span / resolved_steps.min_step_size

    nominal_warn = nominal_count > PROFILE_STEP_PRECHECK_NOMINAL_WARN_THRESHOLD
    dense_warn = dense_count > PROFILE_STEP_PRECHECK_DENSE_WARN_THRESHOLD
    if not nominal_warn and not dense_warn:
        return

    parameter_name = problem.x_names[i_par]
    message = (
        f"Profiling parameter '{parameter_name}' may require many steps "
        f"({nominal_count:.1f} with the default step size, "
        f"up to {dense_count:.1f} with the minimum step size). "
        "Consider increasing the profile step sizes."
    )
    if not options.whole_path:
        message += (
            " This is a bound-based upper estimate; profiling may stop "
            "earlier at the likelihood-ratio threshold."
        )

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
