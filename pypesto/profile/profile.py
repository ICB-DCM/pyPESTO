import logging
import numpy as np
from typing import Callable, Dict, Union

from ..optimize import Optimizer
from ..problem import Problem
from ..result import Result
from .result import ProfilerResult
from .profile_next_guess import next_guess

logger = logging.getLogger(__name__)


class ProfileOptions(dict):
    """
    Options for optimization based profiling.

    Parameters
    ----------
    default_step_size:
        default step size of the profiling routine along the profile path
        (adaptive step lengths algorithms will only use this as a first guess
        and then refine the update)
    min_step_size:
        lower bound for the step size in adaptive methods
    max_step_size:
        upper bound for the step size in adaptive methods
    step_size_factor:
        Adaptive methods recompute the likelihood at the predicted point and
        try to find a good step length by a sort of line search algorithm.
        This factor controls step handling in this line search
    delta_ratio_max:
        maximum allowed drop of the posterior ratio between two profile steps
    ratio_min:
        lower bound for likelihood ratio of the profile, based on inverse
        chi2-distribution.
        The default corresponds to 95% confidence
    reg_points:
        number of profile points used for regression in regression based
        adaptive profile points proposal
    reg_order:
        maximum degree of regression polynomial used in regression based
        adaptive profile points proposal
    magic_factor_obj_value:
        There is this magic factor in the old profiling code which slows down
        profiling at small ratios (must be >= 0 and < 1)
    """

    def __init__(self,
                 default_step_size: float = 0.01,
                 min_step_size: float = 0.001,
                 max_step_size: float = 1.,
                 step_size_factor: float = 1.25,
                 delta_ratio_max: float = 0.1,
                 ratio_min: float = 0.145,
                 reg_points: int = 10,
                 reg_order: int = 4,
                 magic_factor_obj_value: float = 0.5):
        super().__init__()

        self.default_step_size = default_step_size
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.ratio_min = ratio_min
        self.step_size_factor = step_size_factor
        self.delta_ratio_max = delta_ratio_max
        self.reg_points = reg_points
        self.reg_order = reg_order
        self.magic_factor_obj_value = magic_factor_obj_value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def create_instance(
            maybe_options: Union['ProfileOptions', Dict]
    ) -> 'ProfileOptions':
        """
        Returns a valid options object.

        Parameters
        ----------
        maybe_options: ProfileOptions or dict
        """
        if isinstance(maybe_options, ProfileOptions):
            return maybe_options
        options = ProfileOptions(**maybe_options)
        return options


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
    for i_parameter in range(0, problem.dim_full):
        if (profile_index[i_parameter] == 0) or (i_parameter in
                                                 problem.x_fixed_indices):
            continue

        # create an instance of ProfilerResult, which will be appended to the
        # result object, when this profile is finished
        current_profile = result.profile_result.get_current_profile(
            i_parameter)

        # compute profile in descending and ascending direction
        for par_direction in [-1, 1]:
            # flip profile
            current_profile.flip_profile()

            # compute the current profile
            current_profile = walk_along_profile(current_profile,
                                                 problem,
                                                 par_direction,
                                                 optimizer,
                                                 profile_options,
                                                 create_next_guess,
                                                 global_opt,
                                                 i_parameter)

        # add current profile to result.profile_result
        # result.profile_result.add_profile(current_profile, i_parameter)

    # return
    return result


def walk_along_profile(current_profile,
                       problem,
                       par_direction,
                       optimizer,
                       options,
                       create_next_guess,
                       global_opt,
                       i_parameter):
    """
    This is function compute a half-profile

    Parameters
    ----------

    current_profile: pypesto.ProfilerResults
        The profile which should be computed

    problem: pypesto.Problem
        The problem to be solved.

    par_direction: integer
        Indicates profiling direction (+1, -1: ascending, descending)

    optimizer: pypesto.Optimizer
        The optimizer to be used along each profile.

    global_opt: float
        log-posterior value of the global optimum

    options: pypesto.ProfileOptions
        Various options applied to the profile optimization.

    create_next_guess: callable
        Handle of the method which creates the next profile point proposal

    i_parameter: integer
        index for the current parameter
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
        optimizer_result = optimizer.minimize(problem, startpoint, '0')
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
        problem,
        result,
        result_index,
        profile_index,
        profile_list):
    """
    This is function initializes profiling based on a previous optimization.

    Parameters
    ----------

    problem: pypesto.Problem
        The problem to be solved.

    result: pypesto.Result
        A result object to initialize profiling and to append the profiling
        results to. For example, one might append more profiling runs to a
        previous profile, in order to merge these.
        The existence of an optimization result is obligatory.

    result_index: integer
        index from which optimization result profiling should be started

    profile_index: ndarray of integers, optional
        array with parameter indices, whether a profile should
        be computed (1) or not (0)
        Default is all profiles should be computed

    profile_list: integer, optional
        integer which specifies whether a call to the profiler should create
        a new list of profiles (default) or should be added to a specific
        profile list
    """

    # Check, whether an optimization result is existing
    if result.optimize_result is None:
        logger.error(
            "Optimization has to be carried before profiling can be done.")
        return None

    tmp_optimize_result = result.optimize_result.as_list()

    # Check if new profile_list is to be created
    if profile_list is None:
        result.profile_result.create_new_profile_list()

    # get the log-posterior of the global optimum
    global_opt = tmp_optimize_result[0]["fval"]

    # fill the list with optimization results where necessary
    fill_profile_list(result.profile_result,
                      tmp_optimize_result[result_index],
                      profile_index,
                      profile_list,
                      problem.dim_full,
                      global_opt)

    # return the log-posterior of the global optimum (needed in order to
    # compute the log-posterior-ratio)
    return global_opt


def fill_profile_list(
        profile_result,
        optimize_result,
        profile_index,
        profile_list,
        problem_dimension,
        global_opt):
    """
        This is a helper function for initialize_profile

        Parameters
        ----------

        problem: pypesto.Problem
            The problem to be solved.

        profile_result: list of ProfilerResult objects
            A list of profiler result objects

        profile_index: ndarray of integers, optional
            array with parameter indices, whether a profile should
            be computed (1) or not (0)
            Default is all profiles should be computed

        profile_list: integer, optional
            integer which specifies whether a call to the profiler should
            create a new list of profiles (default) or should be added to a
            specific profile list

        problem_dimension: integer
            number of parameters in the unreduced problem

        global_opt: float
            log-posterior at global optimum
        """

    if optimize_result["grad"] is not None:
        gradnorm = np.linalg.norm(optimize_result["grad"])
    else:
        gradnorm = None

    # create blanko profile
    new_profile = ProfilerResult(
        optimize_result["x"],
        np.array([optimize_result["fval"]]),
        np.array([np.exp(global_opt - optimize_result["fval"])]),
        gradnorm,
        optimize_result["exitflag"],
        np.array([0.]),
        np.array([0.]),
        np.array([0]),
        np.array([0]),
        np.array([0]),
        None)

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
                create_new = (profile_result.list[profile_list][i_parameter] is
                              None) and (profile_index[i_parameter] > 0)
                if create_new:
                    profile_result.add_profile(new_profile, i_parameter)
