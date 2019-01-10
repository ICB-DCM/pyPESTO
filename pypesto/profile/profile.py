import logging
import numpy as np
from pypesto import Result
from ..optimize import OptimizeOptions
from .profiler import ProfilerResult
from .profile_next_guess import next_guess

logger = logging.getLogger(__name__)


class ProfileOptions(dict):
    """
    Options for optimization based profiling.

    Parameters
    ----------

    default_step_size: float, optional
        default step size of the profiling routine along the profile path
        (adaptive step lengths algorithms will only use this as a first guess
        and then refine the update)

    ratio_min: float, optional
        lower bound for likelihood ratio of the profile, based on inverse
        chi2-distribution.
        The default corresponds to 95% confidence
    """

    def __init__(self,
                 default_step_size=0.01,
                 min_step_size=0.001,
                 max_step_size=1.,
                 step_size_factor=1.5,
                 delta_ratio_max=0.2,
                 ratio_min=0.145):
        super().__init__()

        self.default_step_size = default_step_size
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.ratio_min = ratio_min
        self.step_size_factor = step_size_factor
        self.delta_ratio_max = delta_ratio_max

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def assert_instance(maybe_options):
        """
        Returns a valid options object.

        Parameters
        ----------

        maybe_options: OptimizeOptions or dict
        """
        if isinstance(maybe_options, ProfileOptions):
            return maybe_options
        options = ProfileOptions(**maybe_options)
        return options


def profile(
        problem,
        result,
        optimizer,
        profile_index=None,
        profile_list=None,
        result_index=0,
        next_guess_method='adaptive_step_order_0',
        profile_options=None,
        optimize_options=None) -> Result:
    """
    This is the main function to call to do parameter profiling.

    Parameters
    ----------

    problem: pypesto.Problem
        The problem to be solved.

    result: pypesto.Result
        A result object to initialize profiling and to append the profiling
        results to. For example, one might append more profiling runs to a
        previous profile, in order to merge these.
        The existence of an optimization result is obligatory.

    optimizer: pypesto.Optimizer
        The optimizer to be used along each profile.

    profile_index: ndarray of integers, optional
        array with parameter indices, whether a profile should
        be computed (1) or not (0)
        Default is all profiles should be computed

    profile_list: integer, optional
        integer which specifies whether a call to the profiler should create
        a new list of profiles (default) or should be added to a specific
        profile list

    result_index: integer, optional
        index from which optimization result profiling should be started
        (default: global optimum, i.e., index = 0)

    next_guess_method: callable, optional
        function handle to a method that creates the next starting point for
        optimization in profiling.

    profile_options: pypesto.ProfileOptions, optional
        Various options applied to the profile optimization.

    optimize_options: pypesto.OptimizeOptions, optional
        Various options applied to the optimizer.
    """

    # profiling indices
    if profile_index is None:
        profile_index = np.ones(problem.dim_full)

    # check profiling options
    if profile_options is None:
        profile_options = ProfileOptions()
    profile_options = ProfileOptions.assert_instance(profile_options)

    # profile startpoint method
    if next_guess_method is None:
        def create_next_guess(x, par_index, par_direction, profile_options,
                              current_profile, problem, global_opt):
            return next_guess(x, par_index, par_direction, profile_options,
                              'fixed_step', current_profile, problem,
                              global_opt)
    elif isinstance(next_guess_method, str):
        def create_next_guess(x, par_index, par_direction, profile_options,
                              current_profile, problem, global_opt):
            return next_guess(x, par_index, par_direction, profile_options,
                              next_guess_method, current_profile, problem,
                              global_opt)
    elif isinstance(next_guess_method, collections.Callable):
        raise Exception('Passing function handles for computation of next '
                        'profiling point is not yet supported.')
    else:
        raise Exception('Unsupported input for create_next_startpoint.')

    # check optimization ptions
    if optimize_options is None:
        optimize_options = OptimizeOptions()
    optimize_options = OptimizeOptions.assert_instance(optimize_options)

    # create the profile result object (retrieve global optimum) ar append to
    #  existing list of profiles
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
        if par_direction is -1:
            stop_profile = (x_now[i_parameter] <= problem.lb_full[[
                i_parameter]]) or (current_profile.ratio_path[-1] <
                                   options.ratio_min)

        if par_direction is 1:
            stop_profile = (x_now[i_parameter] >= problem.ub_full[[
                i_parameter]]) or (current_profile.ratio_path[-1] <
                                   options.ratio_min)

        if stop_profile:
            break

        # compute the new start point for optimization
        (obj_next, x_next) = \
            create_next_guess(x_now, i_parameter, par_direction, options,
                              current_profile, problem, global_opt)

        # fix current profiling parameter to current value and set
        # start point
        problem.fix_parameters(i_parameter, x_next[i_parameter])
        startpoint = np.array([x_next[i] for i in problem.x_free_indices])

        # run optimization
        # IMPORTANT: This optimization will need a proper exception
        # handling (coming soon)
        optimizer_result = optimizer.minimize(problem, startpoint, 0)
        current_profile.append_profile_point(
            optimizer_result.x,
            optimizer_result.fval,
            np.exp(global_opt - optimizer_result.fval),
            np.linalg.norm(optimizer_result.grad[problem.x_free_indices]),
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
        print("Optimization has to be carried before profiling can be done.")
        return None

    tmp_optimize_result = result.optimize_result.as_list()

    # Check if new profile_list is to be created
    if profile_list is None:
        result.profile_result.create_new_profile_list()

    # fill the list with optimization results where necessary
    fill_profile_list(result.profile_result,
                      tmp_optimize_result[result_index],
                      profile_index,
                      profile_list,
                      problem.dim_full)

    # return the log-posterior of the global optimum (needed in order to
    # compute the log-posterior-ratio)
    return tmp_optimize_result[0]["fval"]


def fill_profile_list(
        profile_result,
        optimize_result,
        profile_index,
        profile_list,
        problem_dimension):
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
        """

    # create blanko profile
    new_profile = ProfilerResult(
        optimize_result["x"],
        np.array([optimize_result["fval"]]),
        np.array([1.]),
        np.linalg.norm(optimize_result["grad"]),
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
