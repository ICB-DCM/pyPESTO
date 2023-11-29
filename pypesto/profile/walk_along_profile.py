import logging
from typing import Callable

import numpy as np

from ..C import GRAD
from ..optimize import OptimizeOptions, Optimizer
from ..problem import Problem
from ..result import OptimizerResult, ProfilerResult
from .options import ProfileOptions

logger = logging.getLogger(__name__)


def walk_along_profile(
    current_profile: ProfilerResult,
    problem: Problem,
    par_direction: int,
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
    current_profile:
        The current profile, modified in-place.
    """
    # create variables which are needed during iteration
    stop_profile = False

    # while loop for profiling (will be exited by break command)
    while True:
        # get current position on the profile path
        x_now = current_profile.x_path[:, -1]

        # check if the next profile point needs to be computed
        # ... check bounds
        if par_direction == -1:
            stop_profile = x_now[i_par] <= problem.lb_full[[i_par]]
        elif par_direction == 1:
            stop_profile = x_now[i_par] >= problem.ub_full[[i_par]]
        else:
            raise AssertionError("par_direction must be -1 or 1")

        # ... check likelihood ratio
        if not options.whole_path:
            stop_profile |= current_profile.ratio_path[-1] < options.ratio_min

        if stop_profile:
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
        )

        # fix current profiling parameter to current value and set
        # start point
        problem.fix_parameters(i_par, x_next[i_par])
        startpoint = np.array([x_next[i] for i in problem.x_free_indices])

        # run optimization
        if startpoint.size > 0:
            # number of optimization attempts for the given value of i_par in case
            #  no finite solution is found
            for i_optimize_attempt in range(max_tries):
                optimizer_result = optimizer.minimize(
                    problem=problem,
                    x0=startpoint,
                    id=str(i_optimize_attempt),
                    optimize_options=OptimizeOptions(
                        allow_failed_starts=False
                    ),
                )
                if np.isfinite(optimizer_result.fval):
                    break

                profiled_par_id = problem.x_names[i_par]
                profiled_par_value = startpoint[
                    problem.x_free_indices.index(i_par)
                ]
                logger.warning(
                    f"Optimization at {profiled_par_id}={profiled_par_value} failed."
                )
                # sample a new starting point for another attempt
                #  might be preferable to stay close to the previous point, at least initially,
                #  but for now, we just sample from anywhere within the parameter bounds
                # alternatively, run multi-start optimization
                startpoint = problem.startpoint_method(
                    n_starts=1, problem=problem
                )[0]
            else:
                raise RuntimeError(
                    f"Computing profile point failed. Could not find a finite solution after {max_tries} attempts."
                )
        else:
            # if too many parameters are fixed, there is nothing to do ...
            fval = problem.objective(np.array([]))
            optimizer_result = OptimizerResult(
                id='0',
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
            exitflag=optimizer_result.exitflag,
            n_fval=optimizer_result.n_fval,
            n_grad=optimizer_result.n_grad,
            n_hess=optimizer_result.n_hess,
        )

    # free the profiling parameter again
    problem.unfix_parameters(i_par)

    return current_profile
