import logging
from collections.abc import Iterable
from typing import Callable, Union
from warnings import warn

from ..engine import Engine, SingleCoreEngine
from ..history import HistoryOptions
from ..problem import Problem
from ..result import Result
from ..startpoint import StartpointMethod, to_startpoint_method, uniform
from ..store import autosave
from .optimizer import Optimizer, ScipyOptimizer
from .options import OptimizeOptions
from .task import OptimizerTask
from .util import (
    assign_ids,
    bound_n_starts_from_env,
    postprocess_hdf5_history,
    preprocess_hdf5_history,
)

logger = logging.getLogger(__name__)


def minimize(
    problem: Problem,
    optimizer: Optimizer = None,
    n_starts: int = 100,
    ids: Iterable[str] = None,
    startpoint_method: Union[StartpointMethod, Callable, bool] = None,
    result: Result = None,
    engine: Engine = None,
    progress_bar: bool = None,
    options: OptimizeOptions = None,
    history_options: HistoryOptions = None,
    filename: Union[str, Callable, None] = None,
    overwrite: bool = False,
) -> Result:
    """
    Do multistart optimization.

    Parameters
    ----------
    problem:
        The problem to be solved.
    optimizer:
        The optimizer to be used `n_starts` times.
    n_starts:
        Number of starts of the optimizer.
    ids:
        Ids assigned to the startpoints.
    startpoint_method:
        Method for how to choose start points. ``False`` means the optimizer does
        not require start points, e.g. for the :class:`pypesto.optimize.PyswarmOptimizer`.
        **Deprecated. Use ``problem.startpoint_method`` instead.**
    result:
        A result object to append the optimization results to. For example,
        one might append more runs to a previous optimization. If None,
        a new object is created.
    engine:
        Parallelization engine. Defaults to sequential execution using
        :class:`pypesto.engine.SingleCoreEngine`.
    progress_bar:
        Whether to display a progress bar.
    options:
        Various options applied to the multistart optimization.
    history_options:
        Optimizer history options.
    filename:
        Name of the hdf5 file, where the result will be saved. Default is
        None, which deactivates automatic saving. If set to
        ``Auto`` it will automatically generate a file named
        ``year_month_day_profiling_result.hdf5``.
        Optionally a method, see docs for :func:`pypesto.store.auto.autosave`.
    overwrite:
        Whether to overwrite `result/optimization` in the autosave file
        if it already exists.

    Returns
    -------
    Result object containing the results of all multistarts in
    `result.optimize_result`.
    """
    # optimizer
    if optimizer is None:
        optimizer = ScipyOptimizer()

    # number of starts
    n_starts = bound_n_starts_from_env(n_starts)

    # startpoint method
    if startpoint_method is None:
        if problem.startpoint_method is None:
            startpoint_method = uniform
        else:
            startpoint_method = problem.startpoint_method
    else:
        warn(
            "Passing `startpoint_method` directly is deprecated, "
            "use `problem.startpoint_method` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # convert startpoint method to class instance
    startpoint_method = to_startpoint_method(startpoint_method)

    # check options
    if options is None:
        options = OptimizeOptions()
    options = OptimizeOptions.assert_instance(options)

    # history options
    if history_options is None:
        history_options = HistoryOptions()
    history_options = HistoryOptions.assert_instance(history_options)

    # assign startpoints
    startpoints = startpoint_method(
        n_starts=n_starts,
        problem=problem,
    )

    ids = assign_ids(
        n_starts=n_starts,
        ids=ids,
        result=result,
    )

    # prepare result
    if result is None:
        result = Result(problem)

    # engine
    if engine is None:
        engine = SingleCoreEngine()

    # change to one hdf5 storage file per start if parallel and if hdf5
    history_file = history_options.storage_file
    history_requires_postprocessing = preprocess_hdf5_history(
        history_options, engine
    )

    # define tasks
    tasks = []
    for startpoint, id in zip(startpoints, ids):
        task = OptimizerTask(
            optimizer=optimizer,
            problem=problem,
            x0=startpoint,
            id=id,
            history_options=history_options,
            optimize_options=options,
        )
        tasks.append(task)

    # perform multistart optimization
    ret = engine.execute(tasks, progress_bar=progress_bar)

    # merge hdf5 history files
    if history_requires_postprocessing:
        postprocess_hdf5_history(ret, history_file, history_options)

    # aggregate results
    for optimizer_result in ret:
        result.optimize_result.append(optimizer_result)

    # sort by best fval
    result.optimize_result.sort()

    # if history file provided, set storage file to that one
    if filename == "Auto" and history_file is not None:
        filename = history_file
    autosave(
        filename=filename,
        result=result,
        store_type="optimize",
        overwrite=overwrite,
    )

    return result
