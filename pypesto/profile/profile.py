import copy
import logging
from collections.abc import Iterable
from typing import Callable, Union

import numpy as np

from ..engine import Engine, SingleCoreEngine
from ..optimize import Optimizer
from ..problem import Problem
from ..result import Result
from ..store import autosave
from .options import ProfileOptions
from .profile_next_guess import next_guess
from .task import ProfilerTask
from .util import initialize_profile

logger = logging.getLogger(__name__)


def parameter_profile(
    problem: Problem,
    result: Result,
    optimizer: Optimizer,
    engine: Engine = None,
    profile_index: Iterable[int] = None,
    profile_list: int = None,
    result_index: int = 0,
    next_guess_method: Union[Callable, str] = "adaptive_step_order_1",
    profile_options: ProfileOptions = None,
    progress_bar: bool = None,
    filename: Union[str, Callable, None] = None,
    overwrite: bool = False,
) -> Result:
    """
    Compute parameter profiles.

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
        Defaults to :class:`pypesto.engine.SingleCoreEngine`.
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
        Method that creates the next starting point for optimization in profiling.
        One of the ``update_type`` options supported by
        :func:`pypesto.profile.profile_next_guess.next_guess`.
    profile_options:
        Various options applied to the profile optimization.
        See :class:`pypesto.profile.options.ProfileOptions`.
    progress_bar:
        Whether to display a progress bar.
    filename:
        Name of the hdf5 file, where the result will be saved. Default is
        None, which deactivates automatic saving. If set to
        ``Auto`` it will automatically generate a file named
        ``year_month_day_profiling_result.hdf5``.
        Optionally a method, see docs for :func:`pypesto.store.auto.autosave`.
    overwrite:
        Whether to overwrite `result/profiling` in the autosave file
        if it already exists.

    Returns
    -------
    The profile results are filled into `result.profile_result`.
    """
    # Copy the problem to avoid side effects
    problem = copy.deepcopy(problem)
    # Handling defaults
    # profiling indices
    if profile_index is None:
        profile_index = problem.x_free_indices

    # check profiling options
    if profile_options is None:
        profile_options = ProfileOptions()
    profile_options = ProfileOptions.create_instance(profile_options)
    profile_options.validate()

    # Create a function handle that will be called later to get the next point.
    # This function will be used to generate the initial points of optimization
    # steps in profiling in `walk_along_profile.py`
    if isinstance(next_guess_method, str):

        def create_next_guess(
            x,
            par_index,
            par_direction_,
            profile_options_,
            current_profile_,
            problem_,
            global_opt_,
            min_step_increase_factor_,
            max_step_reduce_factor_,
        ):
            return next_guess(
                x,
                par_index,
                par_direction_,
                profile_options_,
                next_guess_method,
                current_profile_,
                problem_,
                global_opt_,
                min_step_increase_factor_,
                max_step_reduce_factor_,
            )

    elif callable(next_guess_method):
        raise NotImplementedError(
            "Passing function handles for computation of next "
            "profiling point is not yet supported."
        )
    else:
        raise ValueError("Unsupported input for next_guess_method.")

    # create the profile result object (retrieve global optimum) or append to
    # existing list of profiles
    global_opt = initialize_profile(
        problem, result, result_index, profile_index, profile_list
    )
    # if engine==None set SingleCoreEngine() as default
    if engine is None:
        engine = SingleCoreEngine()

    # create Tasks
    tasks = []
    # loop over parameters to create tasks
    for i_par in profile_index:
        # only compute profiles for free parameters
        if i_par in problem.x_fixed_indices:
            # log a warning
            logger.warning(
                f"Parameter {i_par} is fixed and will not be profiled."
            )
            continue

        current_profile = result.profile_result.get_profiler_result(
            i_par=i_par,
            profile_list=profile_list,
        )

        # create two tasks for each parameter: in descending and ascending direction
        for par_direction in [-1, 1]:
            task = ProfilerTask(
                current_profile=copy.deepcopy(current_profile),
                problem=problem,
                optimizer=optimizer,
                options=profile_options,
                create_next_guess=create_next_guess,
                global_opt=global_opt,
                i_par=i_par,
                par_direction=par_direction,
            )
            tasks.append(task)

    # execute the tasks with Engine
    indexed_profiles = engine.execute(tasks, progress_bar=progress_bar)

    # combine the descending and ascending profile halves for each parameter
    paired_profiles = {}
    for indexed_profile in indexed_profiles:
        if indexed_profile["index"] not in paired_profiles:
            paired_profiles[indexed_profile["index"]] = [None, None]
        paired_profiles[indexed_profile["index"]][
            max(indexed_profile["par_direction"], 0)
        ] = indexed_profile["profile"]
    # fill in the ProfilerResults at the right index
    for p_index in paired_profiles:
        # combine the two profile halves before assigning them to the profile result list
        result.profile_result.list[-1][p_index] = combine_profiles_halves(
            *paired_profiles[p_index]
        )

    autosave(
        filename=filename,
        result=result,
        store_type="profile",
        overwrite=overwrite,
    )

    return result


def combine_profiles_halves(profile_desc, profile_asc):
    """
    Combine ascending and descending profile halves into a single profile.

    Parameters
    ----------
    profile_desc : Profile
        The descending half of the profile to be modified and merged.
    profile_asc : Profile
        The ascending half of the profile to be merged.

    Returns
    -------
    Profile
        Combined profile after concatenating paths and summing cumulative values.
    """
    profile_desc.flip_profile()
    profile_desc.x_path = np.concatenate(
        (profile_desc.x_path, profile_asc.x_path), axis=1
    )
    profile_desc.fval_path = np.concatenate(
        (profile_desc.fval_path, profile_asc.fval_path)
    )
    profile_desc.ratio_path = np.concatenate(
        (profile_desc.ratio_path, profile_asc.ratio_path)
    )
    profile_desc.gradnorm_path = np.concatenate(
        (profile_desc.gradnorm_path, profile_asc.gradnorm_path)
    )
    profile_desc.exitflag_path = np.concatenate(
        (profile_desc.exitflag_path, profile_asc.exitflag_path)
    )
    profile_desc.color_path = np.concatenate(
        (profile_desc.color_path, profile_asc.color_path)
    )
    profile_desc.time_path = np.concatenate(
        (profile_desc.time_path, profile_asc.time_path)
    )

    profile_desc.time_total += profile_asc.time_total
    profile_desc.n_fval += profile_asc.n_fval
    profile_desc.n_grad += profile_asc.n_grad
    profile_desc.n_hess += profile_asc.n_hess

    return profile_desc
