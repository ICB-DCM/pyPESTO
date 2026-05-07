import logging
from collections.abc import Callable
from typing import Literal

import numpy as np

from ..C import GRAD
from ..optimize import OptimizeOptions, Optimizer
from ..problem import Problem
from ..result import OptimizerResult, ProfilerResult
from .options import ProfileOptions
from .util import resolve_profile_step_sizes

logger = logging.getLogger(__name__)


def profile_multistart_optimize(
    optimizer: Optimizer,
    problem: Problem,
    startpoint: np.ndarray,
    options: ProfileOptions,
) -> OptimizerResult:
    """
    Perform optimization at one profile point using multiple starts.

    Additional starts are sampled in the reduced parameter space from a
    Gaussian centered at the point suggested by `profile_next_guess`, using
    `options.profile_sampling_sigma` as the standard deviation for each free
    coordinate. The original suggested startpoint is always included unchanged
    as the final start so the helper falls back to the previous single-start
    behavior if all sampled alternatives fail, raise, or are worse.

    Parameters
    ----------
    optimizer:
        The optimizer to use.
    problem:
        The reduced problem with the profiling parameter already fixed.
    startpoint:
        Optimization startpoint suggested by `profile_next_guess`, in reduced
        space.
    options:
        Profile options controlling the number of starts and the Gaussian
        sampling spread around `startpoint`.

    Returns
    -------
    The best finite optimization result across all attempted starts, or the
    result from the original startpoint if all sampled starts fail.
    """
    if options.profile_n_starts == 1:
        return optimizer.minimize(
            problem=problem,
            x0=startpoint,
            id=str(0),
            optimize_options=OptimizeOptions(allow_failed_starts=True),
        )

    sampled_startpoints = np.random.normal(
        loc=startpoint,
        scale=options.profile_sampling_sigma * np.ones_like(startpoint),
        size=(options.profile_n_starts - 1, len(startpoint)),
    )
    sampled_startpoints = np.clip(sampled_startpoints, problem.lb, problem.ub)
    startpoints = np.vstack((sampled_startpoints, startpoint[np.newaxis, :]))

    best_optimizer_result = None
    best_fval = np.inf
    original_start_result = None

    for i_start, candidate_startpoint in enumerate(startpoints):
        optimizer_result = optimizer.minimize(
            problem=problem,
            x0=candidate_startpoint,
            id=str(i_start),
            optimize_options=OptimizeOptions(allow_failed_starts=True),
        )

        if i_start == len(startpoints) - 1:
            original_start_result = optimizer_result

        if (
            np.isfinite(optimizer_result.fval)
            and optimizer_result.fval < best_fval
        ):
            best_fval = optimizer_result.fval
            best_optimizer_result = optimizer_result

    if best_optimizer_result is not None:
        return best_optimizer_result
    return original_start_result


def walk_along_profile(
    current_profile: ProfilerResult,
    problem: Problem,
    par_direction: Literal[1, -1],
    optimizer: Optimizer,
    options: ProfileOptions,
    create_next_guess: Callable,
    global_opt: float,
    i_par: int,
    max_tries: int = 10,
) -> ProfilerResult:
    """
    Compute half a profile.

    Walk ahead in positive direction until some stopping criterion is
    fulfilled. A two-sided profile is obtained by flipping the profile
    direction.

    Parameters
    ----------
    current_profile:
        The profile which should be computed
    problem:
        The problem to be solved.
    par_direction:
        Indicates profiling direction (+1, -1: ascending, descending)
    optimizer:
        The optimizer to be used along each profile.
    global_opt:
        log-posterior value of the global optimum
    options:
        Various options applied to the profile optimization.
    create_next_guess:
        Handle of the method which creates the next profile point proposal
    i_par:
        index for the current parameter
    max_tries:
        If optimization at a given profile point fails, retry optimization
        ``max_tries`` times with randomly sampled starting points.

    Returns
    -------
    The current profile, modified in-place.
    """
    if par_direction not in (-1, 1):
        raise AssertionError("par_direction must be -1 or 1")

    # Warn at most once per non-profiled parameter in each profile half.
    warned_at_bound: set[int] = set()
    resolved_steps = resolve_profile_step_sizes(problem, i_par, options)

    # while loop for profiling (will be exited by break command)
    while True:
        # get current position on the profile path
        x_now = current_profile.x_path[:, -1]
        color_now = current_profile.color_path[-1]

        # check if the next profile point needs to be computed
        # ... check bounds
        if par_direction == -1 and x_now[i_par] <= problem.lb_full[[i_par]]:
            break
        if par_direction == 1 and x_now[i_par] >= problem.ub_full[[i_par]]:
            break

        # ... check likelihood ratio
        if (
            not options.whole_path
            and current_profile.ratio_path[-1] < options.ratio_min
        ):
            break

        optimization_successful = False
        max_step_reduce_factor = 1.0

        while not optimization_successful:
            # Check max_step_size is not reduced below min_step_size
            if (
                resolved_steps.max_step_size * max_step_reduce_factor
                < resolved_steps.min_step_size
            ):
                logger.warning(
                    "Max step size reduced below min step size. "
                    "Setting a lower min step size can help avoid this issue."
                )
                break

            # compute the new start point for optimization
            x_next = create_next_guess(
                x_now,
                i_par,
                par_direction,
                options,
                current_profile,
                problem,
                global_opt,
                1.0,
                max_step_reduce_factor,
            )

            # fix current profiling parameter to current value and set start point
            problem.fix_parameters(i_par, x_next[i_par])
            startpoint = x_next[problem.x_free_indices]

            if startpoint.size > 0:
                optimizer_result = profile_multistart_optimize(
                    optimizer=optimizer,
                    problem=problem,
                    startpoint=startpoint,
                    options=options,
                )

                if np.isfinite(optimizer_result.fval):
                    optimization_successful = True
                    if max_step_reduce_factor == 1.0:
                        # The color of the point is set to black if no changes were made
                        color_next = np.array([0, 0, 0, 1])
                    else:
                        # The color of the point is set to red if the max_step_size was reduced
                        color_next = np.array([1, 0, 0, 1])
                else:
                    max_step_reduce_factor *= 0.5
                    logger.warning(
                        f"Optimization at {problem.x_names[i_par]}={x_next[i_par]} failed. "
                        "Reducing max_step_size to "
                        f"{resolved_steps.max_step_size * max_step_reduce_factor}."
                    )
            else:
                # if too many parameters are fixed, there is nothing to do ...
                fval = problem.objective(np.array([]))
                optimizer_result = OptimizerResult(
                    id="0",
                    x=np.array([]),
                    fval=fval,
                    n_fval=0,
                    n_grad=0,
                    n_res=0,
                    n_hess=0,
                    n_sres=0,
                    x0=np.array([]),
                    fval0=fval,
                    time=0,
                )
                optimizer_result.update_to_full(problem=problem)
                optimization_successful = True
                color_next = np.concatenate((color_now[:3], [0.3]))

        if not optimization_successful:
            # Cannot optimize successfully by reducing max_step_size
            # Let's try to optimize by increasing min_step_size
            logger.warning(
                f"Failing to optimize at {problem.x_names[i_par]}={x_next[i_par]} after reducing max_step_size."
                f"Trying to increase min_step_size."
            )
            min_step_increase_factor = 1.25
        while not optimization_successful:
            # Check min_step_size is not increased above max_step_size
            if (
                resolved_steps.min_step_size * min_step_increase_factor
                > resolved_steps.max_step_size
            ):
                logger.warning(
                    "Min step size increased above max step size. "
                    "Setting a higher max step size can help avoid this issue."
                )
                break

            # compute the new start point for optimization
            x_next = create_next_guess(
                x_now,
                i_par,
                par_direction,
                options,
                current_profile,
                problem,
                global_opt,
                min_step_increase_factor,
                1.0,
            )

            # fix current profiling parameter to current value and set start point
            problem.fix_parameters(i_par, x_next[i_par])
            startpoint = x_next[problem.x_free_indices]

            optimizer_result = profile_multistart_optimize(
                optimizer=optimizer,
                problem=problem,
                startpoint=startpoint,
                options=options,
            )

            if np.isfinite(optimizer_result.fval):
                optimization_successful = True
                # The color of the point is set to blue if the min_step_size was increased
                color_next = np.array([0, 0, 1, 1])
            else:
                min_step_increase_factor *= 1.25
                logger.warning(
                    f"Optimization at {problem.x_names[i_par]}={x_next[i_par]} failed. "
                    "Increasing min_step_size to "
                    f"{resolved_steps.min_step_size * min_step_increase_factor}."
                )

        if not optimization_successful:
            # Cannot optimize successfully by reducing max_step_size or increasing min_step_size
            # sample a new starting point for another attempt for max_tries times
            logger.warning(
                f"Failing to optimize at {problem.x_names[i_par]}={x_next[i_par]} after reducing max_step_size."
                f"Trying to sample {max_tries} new starting points."
            )

            x_next = create_next_guess(
                x_now,
                i_par,
                par_direction,
                options,
                current_profile,
                problem,
                global_opt,
                1.0,
                1.0,
            )

            problem.fix_parameters(i_par, x_next[i_par])

            for i_optimize_attempt in range(max_tries):
                startpoint = problem.startpoint_method(
                    n_starts=1, problem=problem
                )[0]

                optimizer_result = optimizer.minimize(
                    problem=problem,
                    x0=startpoint,
                    id=str(i_optimize_attempt),
                    optimize_options=OptimizeOptions(
                        allow_failed_starts=False
                    ),
                )
                if np.isfinite(optimizer_result.fval):
                    # The color of the point is set to green if the parameter was resampled
                    color_next = np.array([0, 1, 0, 1])
                    break

                logger.warning(
                    f"Optimization at {problem.x_names[i_par]}={x_next[i_par]} failed."
                )
            else:
                raise RuntimeError(
                    f"Computing profile point failed. Could not find a finite solution after {max_tries} attempts."
                )

        logger.debug(
            f"Optimization successful for {problem.x_names[i_par]}={x_next[i_par]:.4f}. "
            f"Start fval {problem.objective(x_next[problem.x_free_indices]):.6f}, end fval {optimizer_result.fval:.6f}."
        )

        for k, j_par in enumerate(problem.x_free_indices):
            x_j = optimizer_result.x[j_par]
            lb_j = problem.lb[k]
            ub_j = problem.ub[k]
            at_lb = abs(x_j - lb_j) <= 1e-8
            at_ub = abs(x_j - ub_j) <= 1e-8
            if (at_lb or at_ub) and j_par not in warned_at_bound:
                warned_at_bound.add(j_par)
                bound_val = lb_j if at_lb else ub_j
                logger.warning(
                    f"Parameter '{problem.x_names[j_par]}' hit its "
                    f"{'lower' if at_lb else 'upper'} bound "
                    f"({bound_val:.4g}) while profiling "
                    f"'{problem.x_names[i_par]}'. "
                    "The profile may be constrained near this region."
                )

        if optimizer_result[GRAD] is not None:
            gradnorm = np.linalg.norm(
                optimizer_result[GRAD][problem.x_free_indices]
            )
        else:
            gradnorm = np.nan

        current_profile.append_profile_point(
            x=optimizer_result.x,
            fval=optimizer_result.fval,
            ratio=np.exp(global_opt - optimizer_result.fval),
            gradnorm=gradnorm,
            time=optimizer_result.time,
            color=color_next,
            exitflag=optimizer_result.exitflag,
            n_fval=optimizer_result.n_fval,
            n_grad=optimizer_result.n_grad,
            n_hess=optimizer_result.n_hess,
        )

    # free the profiling parameter again
    problem.unfix_parameters(i_par)

    return current_profile
