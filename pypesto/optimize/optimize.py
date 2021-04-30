import logging
from typing import Callable, Iterable, Union

from ..engine import Engine, SingleCoreEngine
from ..objective import HistoryOptions
from ..problem import Problem
from ..result import Result
from ..startpoint import assign_startpoints, uniform
from .optimizer import Optimizer, ScipyOptimizer
from .options import OptimizeOptions
from .task import OptimizerTask
from .util import check_hdf5_mp, fill_hdf5_file

logger = logging.getLogger(__name__)


def minimize(
        problem: Problem,
        optimizer: Optimizer = None,
        n_starts: int = 100,
        ids: Iterable[str] = None,
        startpoint_method: Union[Callable, bool] = None,
        result: Result = None,
        engine: Engine = None,
        progress_bar: bool = True,
        options: OptimizeOptions = None,
        history_options: HistoryOptions = None,
) -> Result:
    """
    This is the main function to call to do multistart optimization.

    Parameters
    ----------
    problem:
        The problem to be solved.
    optimizer:
        The optimizer to be used n_starts times.
    n_starts:
        Number of starts of the optimizer.
    ids:
        Ids assigned to the startpoints.
    startpoint_method:
        Method for how to choose start points. False means the optimizer does
        not require start points, e.g. for the 'PyswarmOptimizer'.
    result:
        A result object to append the optimization results to. For example,
        one might append more runs to a previous optimization. If None,
        a new object is created.
    engine:
        Parallelization engine. Defaults to sequential execution on a
        SingleCoreEngine.
    progress_bar:
        Whether to display a progress bar.
    options:
        Various options applied to the multistart optimization.
    history_options:
        Optimizer history options.

    Returns
    -------
    result:
        Result object containing the results of all multistarts in
        `result.optimize_result`.
    """

    # optimizer
    if optimizer is None:
        optimizer = ScipyOptimizer()

    # startpoint method
    if (startpoint_method is not None) \
            and (problem.startpoint_method is not None):
        raise Warning('Problem.startpoint_method will be ignored. Start '
                      'points will be generated using the startpoint method '
                      'given as an argument to the minimize function.')
    elif problem.startpoint_method is not None:
        startpoint_method = problem.startpoint_method
    elif startpoint_method is None:
        startpoint_method = uniform

    # check options
    if options is None:
        options = OptimizeOptions()
    options = OptimizeOptions.assert_instance(options)

    if history_options is None:
        history_options = HistoryOptions()
    history_options = HistoryOptions.assert_instance(history_options)

    # assign startpoints
    startpoints = assign_startpoints(
        n_starts=n_starts, startpoint_method=startpoint_method,
        problem=problem, startpoint_resample=options.startpoint_resample)

    if ids is None:
        ids = [str(j) for j in range(n_starts)]
    if len(ids) != n_starts:
        raise AssertionError("Number of starts and ids must coincide.")

    # prepare result
    if result is None:
        result = Result(problem)

    # engine
    if engine is None:
        engine = SingleCoreEngine()

    # define tasks
    tasks = []
    filename = None
    if history_options.storage_file is not None and \
            history_options.storage_file.endswith(('.h5', '.hdf5')):
        filename = check_hdf5_mp(history_options, engine)

    for startpoint, id in zip(startpoints, ids):
        task = OptimizerTask(
            optimizer=optimizer, problem=problem, x0=startpoint, id=id,
            options=options, history_options=history_options)
        tasks.append(task)

    # do multistart optimization
    ret = engine.execute(tasks, progress_bar=progress_bar)

    if filename is not None:
        fill_hdf5_file(ret, filename)

    # aggregate results
    for optimizer_result in ret:
        result.optimize_result.append(optimizer_result)

    # sort by best fval
    result.optimize_result.sort()

    return result
