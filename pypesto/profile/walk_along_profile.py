import logging
import numpy as np
from typing import Callable

from ..objective.constants import GRAD
from ..optimize import Optimizer, OptimizerResult
from ..problem import Problem
from .result import ProfilerResult
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
        i_par: int
) -> ProfilerResult:
    """
    This function computes half a profile, by walking ahead in positive
    direction until some stopping criterion is fulfilled. A two-sided profile
    is obtained by flipping the profile direction.

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
        if par_direction == -1:
            stop_profile = (x_now[i_par] <= problem.lb_full[[i_par]]) \
                or (current_profile.ratio_path[-1] < options.ratio_min)

        if par_direction == 1:
            stop_profile = (x_now[i_par] >= problem.ub_full[[i_par]]) \
                or (current_profile.ratio_path[-1] < options.ratio_min)

        if stop_profile:
            break

        # compute the new start point for optimization
        x_next = create_next_guess(x_now, i_par, par_direction,
                                   options, current_profile, problem,
                                   global_opt)

        # fix current profiling parameter to current value and set
        # start point
        problem.fix_parameters(i_par, x_next[i_par])
        startpoint = np.array([x_next[i] for i in problem.x_free_indices])

        # run optimization
        # IMPORTANT: This optimization will need a proper exception
        # handling (coming soon)
        if startpoint.size > 0:
            optimizer_result = optimizer.minimize(problem, startpoint, '0',
                                                  allow_failed_starts=False)
        else:
            # if too many parameters are fixed, there is nothing to do ...
            fval = problem.objective([])
            optimizer_result = OptimizerResult(
                id='0', x=np.array([]), fval=fval, n_fval=0, n_grad=0, n_res=0,
                n_hess=0, n_sres=0, x0=np.array([]), fval0=fval, time=0)
            optimizer_result.update_to_full(problem=problem)

        if optimizer_result[GRAD] is not None:
            gradnorm = np.linalg.norm(optimizer_result[GRAD][
                                      problem.x_free_indices])
        else:
            gradnorm = None

        current_profile.append_profile_point(
            x=optimizer_result.x,
            fval=optimizer_result.fval,
            ratio=np.exp(global_opt - optimizer_result.fval),
            gradnorm=gradnorm,
            time=optimizer_result.time,
            exitflag=optimizer_result.exitflag,
            n_fval=optimizer_result.n_fval,
            n_grad=optimizer_result.n_grad,
            n_hess=optimizer_result.n_hess)

    # free the profiling parameter again
    problem.unfix_parameters(i_par)

    return current_profile
