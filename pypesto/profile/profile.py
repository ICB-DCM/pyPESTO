import logging
from typing import Callable, Union, Iterable

from ..engine import Engine, SingleCoreEngine
from ..optimize import Optimizer
from ..problem import Problem
from ..result import Result
from .profile_next_guess import next_guess
from .options import ProfileOptions
from .util import initialize_profile
from .task import ProfilerTask

logger = logging.getLogger(__name__)


def parameter_profile(
        problem: Problem,
        result: Result,
        optimizer: Optimizer,
        engine: Engine = None,
        profile_index: Iterable[int] = None,
        profile_list: int = None,
        result_index: int = 0,
        next_guess_method: Union[Callable, str] = 'adaptive_step_regression',
        profile_options: ProfileOptions = None,
        progress_bar: bool = True
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
    engine:
        The engine to be used.
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
    progress_bar:
        Whether to display a progress bar.

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
    # if engine==None set SingleCoreEngine() as default
    if engine is None:
        engine = SingleCoreEngine()

    # create Tasks
    tasks = []
    # loop over parameters to create tasks
    for i_par in profile_index:
        # only compute profiles for free parameters
        if i_par in problem.x_fixed_indices:
            continue

        current_profile = result.profile_result.get_profiler_result(
            i_par=i_par, profile_list=profile_list)

        task = ProfilerTask(
            current_profile=current_profile,
            problem=problem,
            optimizer=optimizer,
            options=profile_options,
            create_next_guess=create_next_guess,
            global_opt=global_opt,
            i_par=i_par
        )
        tasks.append(task)

    # execute the tasks with Engine
    indexed_profiles = engine.execute(tasks, progress_bar=progress_bar)

    # fill in the ProfilerResults at the right index
    for indexed_profile in indexed_profiles:
        result.profile_result.list[-1][indexed_profile['index']] = \
            indexed_profile['profile']

    return result
