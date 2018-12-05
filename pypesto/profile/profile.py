import logging
import numpy as np
from pypesto import Result
from ..optimize import OptimizeOptions
from .profiler import ProfilerResult
from .profile_startpoint import fixed_step

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
                 ratio_min=0.145):
        super().__init__()

        self.default_step_size = default_step_size
        self.ratio_min = ratio_min

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
        result_index=1,
        create_profile_startpoint=None,
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

    result_index: integer, optional
        index from which optimization result profiling should be started
        (default: global optimum, i.e., index = 1)

    create_profile_startpoint: callable, optional
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

    # profile startpoint method
    if create_profile_startpoint is None:
        def create_next_startpoint(x, par_index, par_direction):
            return fixed_step(x, par_index, par_direction,
                              step_size=profile_options.default_step_size)

    # check profiling options
    if profile_options is None:
        profile_options = ProfileOptions()
    profile_options = ProfileOptions.assert_instance(profile_options)

    # check optimization ptions
    if optimize_options is None:
        optimize_options = OptimizeOptions()
    optimize_options = OptimizeOptions.assert_instance(optimize_options)

    # create the profile result object and retrieve global optimum
    global_opt = initialize_profile(problem, result, result_index)

    # loop over parameters for profiling
    for i_parameter in range(0, problem.dim_full):
        if (profile_index[i_parameter] == 0) | \
                (i_parameter in problem.x_fixed_indices):
            continue

        # create an instance of ProfilerResult, which will be appended to the
        #  result object, when this profile is finished
        current_profile = \
            result.profile_result.get_current_profile(i_parameter)
        lb_old = None

        # compute profile in descending and ascending direction
        for par_direction in [-1, 1]:

            # flip profile
            current_profile.flip_profile()

            # while loop for profiling (will be exited by break command)
            while True:
                # get current position on the profile path
                x_now = current_profile.x_path[:, -1]

                # check if the next profile point needs to be computed
                if par_direction is -1:
                    stop_profile = (x_now[i_parameter] <= problem.lb[
                        [i_parameter]]) | \
                                   (current_profile.ratio_path[
                                        -1] < profile_options.ratio_min)

                if par_direction is 1:
                    stop_profile = (x_now[i_parameter] >= problem.ub[
                        [i_parameter]]) | \
                                   (current_profile.ratio_path[
                                        -1] < profile_options.ratio_min)

                if stop_profile:
                    break

                # compute the new start point for optimization
                x_next = create_next_startpoint(x_now, i_parameter,
                                                par_direction)

                # check whether the next point is maybe outside the bounds
                # and correct it
                if par_direction is -1:
                    x_next[i_parameter] = np.max(
                        [x_next[i_parameter], problem.lb[i_parameter]])
                else:
                    x_next[i_parameter] = np.min(
                        [x_next[i_parameter], problem.ub[i_parameter]])

                # fix current profiling parameter to current value and set
                # start point (retrieve old bounds, if necessary, in order to
                # re-adapt them when this parameter is freed again)
                if lb_old is None:
                    (lb_old, ub_old) = \
                        problem.fix_parameters(i_parameter,x_next[i_parameter])
                else:
                    problem.fix_parameters(i_parameter, x_next[i_parameter])
                startpoint = np.array(
                    [x_next[i] for i in problem.x_free_indices])

                # run optimization
                # IMPORTANT: This optimization will need a proper exception
                # handling (coming soon)
                optimizer_result = optimizer.minimize(problem, startpoint, 0)
                current_profile.append_profile_point(
                    optimizer_result.x,
                    optimizer_result.fval,
                    np.exp(global_opt - optimizer_result.fval),
                    np.linalg.norm(
                        optimizer_result.grad[problem.x_free_indices]),
                    optimizer_result.exitflag,
                    optimizer_result.time,
                    optimizer_result.n_fval,
                    optimizer_result.n_grad,
                    optimizer_result.n_hess)

        # free the profiling parameter again
        problem.unfix_parameters(i_parameter, lb_old, ub_old)

        # add current profile to result.profile_result
        result.profile_result.add_profile(current_profile, i_parameter)

    # return
    return result


def initialize_profile(
        problem,
        result,
        result_index):
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
    """

    # check, whether an optimization result is existing
    if result.optimize_result is None:
        print("Optimization has to be carried before profiling can be done.")
        return None

    # create the ProfilerResult beased on the optimization result
    for iParameter in range(0, problem.dim_full):
        tmp_optimize_result = result.optimize_result.as_list()
        result.profile_result.create_new_profile(
            ProfilerResult(tmp_optimize_result[result_index]["x"],
                           tmp_optimize_result[result_index]["fval"],
                           np.array([1.]),
                           np.linalg.norm(
                               tmp_optimize_result[result_index]["grad"]),
                           tmp_optimize_result[result_index]["exitflag"],
                           np.array([0.]),
                           np.array([0.]),
                           np.array([0]),
                           np.array([0]),
                           np.array([0]),
                           None))

    # return the log-posterior of the global optimum (needed in order to
    # compute the log-posterior-ratio)
    return tmp_optimize_result[0]["fval"]
