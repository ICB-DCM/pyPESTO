import logging
import numpy as np
from typing import Callable, Union, Iterable

from ..objective.constants import GRAD
from ..optimize import Optimizer, OptimizerResult
from ..problem import Problem
from ..result import Result
from .result import ProfilerResult
from .profile_next_guess import next_guess
from .options import ProfileOptions
from .util import initialize_profile

logger = logging.getLogger(__name__)


def parameter_profile(
        problem: Problem,
        result: Result,
        optimizer: Optimizer,
        profile_index: Iterable[int] = None,
        profile_list: int = None,
        result_index: int = 0,
        next_guess_method: Union[Callable, str] = 'adaptive_step_regression',
        profile_options: ProfileOptions = None
) -> Result:
    """
    This is the main function to call to do parameter profiling.

    Parameters
    ----------
    problem:
        The problem to be solved.
    result:
        A result object to initialize profiling and to append the profiling
        results to. For example, one might append more profiling runs to a
        previous profile, in order to merge these.
        The existence of an optimization result is obligatory.
    optimizer:
        The optimizer to be used along each profile.
    profile_index:
        List with the parameter indices to be profiled
        (by default all free indices).
    profile_list:
        Integer which specifies whether a call to the profiler should create
        a new list of profiles (default) or should be added to a specific
        profile list.
    result_index:
        Index from which optimization result profiling should be started
        (default: global optimum, i.e., index = 0).
    next_guess_method:
        Function handle to a method that creates the next starting point for
        optimization in profiling.
    profile_options:
        Various options applied to the profile optimization.

    Returns
    -------
    result:
        The profile results are filled into `result.profile_result`.
    """
    # Handling defaults
    # profiling indices
    if profile_index is None:
        profile_index = problem.x_free_indices

    # check profiling options
    if profile_options is None:
        profile_options = ProfileOptions()
    profile_options = ProfileOptions.create_instance(profile_options)

    # create a function handle that will be called later to get the next point
    if isinstance(next_guess_method, str):
        def create_next_guess(x, par_index, par_direction_, profile_options_,
                              current_profile_, problem_, global_opt_):
            return next_guess(x, par_index, par_direction_, profile_options_,
                              next_guess_method, current_profile_, problem_,
                              global_opt_)
    elif callable(next_guess_method):
        raise Exception('Passing function handles for computation of next '
                        'profiling point is not yet supported.')
    else:
        raise Exception('Unsupported input for next_guess_method.')

    # create the profile result object (retrieve global optimum) or append to
    # existing list of profiles
    global_opt = initialize_profile(problem, result, result_index,
                                    profile_index, profile_list)

    # loop over parameters for profiling
    for i_par in profile_index:
        # only compute profiles for free parameters
        if i_par in problem.x_fixed_indices:
            continue

        # create an instance of ProfilerResult, which will be appended to the
        # result object, when this profile is finished
        current_profile = result.profile_result.get_profiler_result(
            i_par=i_par, profile_list=profile_list)

        # compute profile in descending and ascending direction
        for par_direction in [-1, 1]:
            # flip profile
            current_profile.flip_profile()

            # compute the current profile
            current_profile = walk_along_profile(
                current_profile=current_profile,
                problem=problem,
                par_direction=par_direction,
                optimizer=optimizer,
                options=profile_options,
                create_next_guess=create_next_guess,
                global_opt=global_opt,
                i_par=i_par)

    return result


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
