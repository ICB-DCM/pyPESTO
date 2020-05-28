import logging
import numpy as np
from typing import Any, Callable, Dict

from ..optimize import Optimizer
from ..problem import Problem
from ..result import Result, ProfileResult
from .result import ProfilerResult
from .profile_next_guess import next_guess
from .options import ProfileOptions

logger = logging.getLogger(__name__)


def parameter_profile(
        problem: Problem,
        result: Result,
        optimizer: Optimizer,
        profile_index: np.ndarray = None,
        profile_list: int = None,
        result_index: int = 0,
        next_guess_method: Callable = None,
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
        array with parameter indices, whether a profile should
        be computed (1) or not (0)
        Default is all profiles should be computed
    profile_list:
        integer which specifies whether a call to the profiler should create
        a new list of profiles (default) or should be added to a specific
        profile list
    result_index:
        index from which optimization result profiling should be started
        (default: global optimum, i.e., index = 0)
    next_guess_method:
        function handle to a method that creates the next starting point for
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
        profile_index = np.ones(problem.dim_full)
        profile_index[problem.x_fixed_indices] = 0

    # check profiling options
    if profile_options is None:
        profile_options = ProfileOptions()
    profile_options = ProfileOptions.create_instance(profile_options)

    # profile startpoint method
    if next_guess_method is None:
        next_guess_method = 'adaptive_step_regression'

    # create a function handle that will be called later to get the next point
    if isinstance(next_guess_method, str):
        def create_next_guess(x, par_index, par_direction, profile_options,
                              current_profile, problem, global_opt):
            return next_guess(x, par_index, par_direction, profile_options,
                              next_guess_method, current_profile, problem,
                              global_opt)
    elif callable(next_guess_method):
        raise Exception('Passing function handles for computation of next '
                        'profiling point is not yet supported.')
    else:
        raise Exception('Unsupported input for next_guess_method.')

    # create the profile result object (retrieve global optimum) ar append to
    # existing list of profiles
    global_opt = initialize_profile(problem, result, result_index,
                                    profile_index, profile_list)

    # loop over parameters for profiling
    for i_par in range(0, problem.dim_full):
        if profile_index[i_par] == 0 or i_par in problem.x_fixed_indices:
            # not requested or fixed -> compute no profile
            continue

        # create an instance of ProfilerResult, which will be appended to the
        # result object, when this profile is finished
        current_profile = result.profile_result.get_current_profile(
            i_par)

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
                i_parameter=i_par)

        # add current profile to result.profile_result
        # result.profile_result.add_profile(current_profile, i_parameter)

    return result


def walk_along_profile(
        current_profile: ProfilerResult,
        problem: Problem,
        par_direction: int,
        optimizer: Optimizer,
        options: ProfileOptions,
        create_next_guess: Callable,
        global_opt: float,
        i_parameter: int
) -> ProfilerResult:
    """
    This is function compute a half-profile

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
    i_parameter:
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
            stop_profile = (x_now[i_parameter] <= problem.lb_full[[
                i_parameter]]) or (current_profile.ratio_path[-1] <
                                   options.ratio_min)

        if par_direction == 1:
            stop_profile = (x_now[i_parameter] >= problem.ub_full[[
                i_parameter]]) or (current_profile.ratio_path[-1] <
                                   options.ratio_min)

        if stop_profile:
            break

        # compute the new start point for optimization
        x_next = create_next_guess(x_now, i_parameter, par_direction,
                                   options, current_profile, problem,
                                   global_opt)

        # fix current profiling parameter to current value and set
        # start point
        problem.fix_parameters(i_parameter, x_next[i_parameter])
        startpoint = np.array([x_next[i] for i in problem.x_free_indices])

        # run optimization
        # IMPORTANT: This optimization will need a proper exception
        # handling (coming soon)
        optimizer_result = optimizer.minimize(problem, startpoint, '0',
                                              allow_failed_starts=False)
        if optimizer_result["grad"] is not None:
            gradnorm = np.linalg.norm(optimizer_result["grad"][
                                      problem.x_free_indices])
        else:
            gradnorm = None

        current_profile.append_profile_point(
            optimizer_result.x,
            optimizer_result.fval,
            np.exp(global_opt - optimizer_result.fval),
            gradnorm,
            optimizer_result.exitflag,
            optimizer_result.time,
            optimizer_result.n_fval,
            optimizer_result.n_grad,
            optimizer_result.n_hess)

    # free the profiling parameter again
    problem.unfix_parameters(i_parameter)

    return current_profile


def initialize_profile(
        problem: Problem,
        result: Result,
        result_index: int,
        profile_index: np.ndarray,
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
            "Optimization has to be carried before profiling can be done.")

    tmp_optimize_result = result.optimize_result.as_list()

    # Check if new profile_list is to be created
    if profile_list is None:
        result.profile_result.create_new_profile_list()

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
        profile_index: np.ndarray,
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

    if optimizer_result["grad"] is not None:
        gradnorm = np.linalg.norm(optimizer_result["grad"])
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
            if profile_index[i_parameter] > 0:
                # Should we create a profile for this index?
                profile_result.create_new_profile(new_profile)
            else:
                # if no profile should be computed for this parameter
                profile_result.create_new_profile()

    else:
        for i_parameter in range(0, problem_dimension):
            # We append to an existing list
            if profile_index[i_parameter] > 0:
                # Do we have to create a new profile?
                create_new = (profile_result.list[profile_list][i_parameter]
                              is None) and (profile_index[i_parameter] > 0)
                if create_new:
                    profile_result.add_profile(new_profile, i_parameter)
