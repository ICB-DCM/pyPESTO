import logging
from pypesto import Result
from ..optimize import OptimizerResult, OptimizeOptions, minimize
from .profiler import ProfilerResult
from .profile_startpoint import simple_step


logger = logging.getLogger(__name__)


class ProfileOptions(dict):
    """
    Options for optimization based profiling.

    Parameters
    ----------

    next_profile_startpoint: function handle, optional
        function which creates the next startpoint for optimization of the profile from the profile history
    """

    def __init__(self):
        super().__init__()

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

    profile_index: integer array, optional
        array with parameter indices, whether a profile should
        be computed (1) or not (0)

    result_index: integer, optional
        index, from which optimization result profiling should be started
        (default: global optimum, i.e., index = 1)

    create_profile_startpoint: callable, optional
        Method for how to create the next start point for profile optimization.

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
        create_profile_startpoint = simple_step

    # check profiling options
    if profile_options is None:
        profile_options = ProfileOptions()
    profile_options = ProfileOptions.assert_instance(profile_options)

    # check optimization ptions
    if optimize_options is None:
        optimize_options = OptimizeOptions()
    optimize_options = OptimizeOptions.assert_instance(optimize_options)

    # assign startpoints
    profile_result = initialize_profile(problem, result, result_index)

    # do multistart optimization
    for j_start in range(0, n_starts):
        startpoint = startpoints[j_start, :]

        # apply optimizer
        try:
            optimizer_result = optimizer.minimize(problem, startpoint, j_start)
        except Exception as err:
            if options.allow_failed_starts:
                optimizer_result = handle_exception(
                    problem.objective, startpoint, j_start, err)
            else:
                raise

        # append to result
        result.optimize_result.append(profile_result)

    # sort by best fval
    result.optimize_result.sort()

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
           The index, starting from which optimization result profiling
           should be carried out
       """
    if result.optimize_result is None:
        print("Optimization has to be carried before profiling can be done.")
        return None

    current_profile = len(result.profile_result)

    for iParameter in range(0, problem.dim_full):
        tmp_result = {"x_path": result.optimize_result[result_index].x,
                 "fval_path": result.optimize_result[result_index].fval,
                 "ratio_path": np.array([1]),
                 "gradnorm_path": np.linalg.norm(result.optimize_result[result_index].grad),
                 "exitflag_path": result.optimize_result[result_index].exitflag,
                 "time_path": np.array([0]),
                 "time_total": np.array([0]),
                 "n_fval": np.array([0]),
                 "n_grad": np.array([0]),
                 "n_hess": np.array([0]),
                 "message": None}
        result.profile_result[current_profile].append(ProfileResult(tmp_result))

