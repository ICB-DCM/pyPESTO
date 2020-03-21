import logging
from typing import Callable, Union
import numpy as np

from ..engine import OptimizerTask, Engine, SingleCoreEngine
from ..objective import Objective
from ..problem import Problem
from ..result import Result
from ..startpoint import assign_startpoints, uniform
from .optimizer import (
    OptimizerResult, recover_result, Optimizer, ScipyOptimizer)
from .options import OptimizeOptions


logger = logging.getLogger(__name__)


def minimize(
        problem: Problem,
        optimizer: Optimizer = None,
        n_starts: int = 100,
        startpoint_method: Union[Callable, bool] = None,
        result: Result = None,
        engine: Engine = None,
        options: OptimizeOptions = None
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
    startpoint_method:
        Method for how to choose start points. False means the optimizer does
        not require start points, e.g. 'pso' method in 'GlobalOptimizer'
    result:
        A result object to append the optimization results to. For example,
        one might append more runs to a previous optimization. If None,
        a new object is created.
    engine:
        Parallelization engine. Defaults to sequential execution on a
        SingleCoreEngine.
    options:
        Various options applied to the multistart optimization.

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
    if startpoint_method is None:
        startpoint_method = uniform

    # check options
    if options is None:
        options = OptimizeOptions()
    options = OptimizeOptions.assert_instance(options)

    # assign startpoints
    startpoints = assign_startpoints(n_starts, startpoint_method,
                                     problem, options)

    # prepare result
    if result is None:
        result = Result(problem)

    # engine
    if engine is None:
        engine = SingleCoreEngine()

    # define tasks
    tasks = []
    for j_start in range(0, n_starts):
        startpoint = startpoints[j_start, :]
        task = OptimizerTask(optimizer, problem, startpoint, j_start,
                             options, handle_exception)
        tasks.append(task)

    # do multistart optimization
    ret = engine.execute(tasks)

    # aggregate results
    for optimizer_result in ret:
        result.optimize_result.append(optimizer_result)

    # sort by best fval
    result.optimize_result.sort()

    return result


def handle_exception(
        objective: Objective,
        startpoint: np.ndarray,
        j_start: int,
        err: Exception
) -> OptimizerResult:
    """
    Handle exception by creating a dummy pypesto.OptimizerResult.
    """
    logger.error(('start ' + str(j_start) + ' failed: {0}').format(err))
    optimizer_result = recover_result(objective, startpoint, err)
    return optimizer_result
