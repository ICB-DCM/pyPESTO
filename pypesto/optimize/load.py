"""Load/reconstitute results from history."""

import logging
import os

import h5py
import numpy as np

import pypesto

from ..C import FVAL, GRAD, HESS, RES, SRES, X
from ..objective import (
    CsvHistory,
    Hdf5History,
    HistoryOptions,
    OptimizerHistory,
)
from ..problem import Problem
from ..result import OptimizeResult, OptimizerResult, Result
from .options import OptimizeOptions

logger = logging.getLogger(__name__)

EXITFLAG_LOADED_FROM_FILE = -99


def fill_result_from_history(
    result: OptimizerResult,
    optimizer_history: OptimizerHistory,
    optimize_options: OptimizeOptions = None,
) -> OptimizerResult:
    """Overwrite some values in the result object with values in the history.

    Parameters
    ----------
    result: Result as reported from the used optimizer.
    optimizer_history: History of function values recorded by the objective.
    optimize_options: Options on e.g. how to override.

    Returns
    -------
    result: The in-place modified result.
    """
    if optimize_options is None:
        optimize_options = OptimizeOptions()

    # logging
    #  function values
    history_fval, result_fval = optimizer_history.fval_min, result.fval
    fval_exist = history_fval is not None and result_fval is not None
    fval_match = fval_exist and np.isclose(history_fval, result_fval)
    if fval_exist and not fval_match:
        logger.debug(
            f"Minimal function value mismatch: history {history_fval:8e}, "
            f"result {result_fval:8e}"
        )
    #  parameters
    history_x, result_x = optimizer_history.x_min, result.x
    x_exist = history_x is not None and result_x is not None
    x_match = x_exist and np.allclose(history_x, result_x)
    if x_exist and not x_match:
        logger.debug(
            f"Optimal parameter mismatch: history {history_x}, "
            f"result {result_x}"
        )

    # counters
    # we only use our own counters here as optimizers may report differently
    for key in (FVAL, GRAD, HESS, RES, SRES):
        setattr(
            result, f"n_{key}", getattr(optimizer_history.history, f"n_{key}")
        )

    # initial values
    result.x0 = optimizer_history.x0
    result.fval0 = optimizer_history.fval0

    # trace
    result.history = optimizer_history.history

    # if optimizer is trusted, don't overwrite/complement optimal point
    if not optimize_options.history_beats_optimizer:
        return result

    if isinstance(optimizer_history.history, Hdf5History):
        if (message := optimizer_history.history.message) is not None:
            result.message = message
        if (exitflag := optimizer_history.history.exitflag) is not None:
            result.exitflag = exitflag

    # optimal point
    for key in (X, FVAL, GRAD, HESS, RES, SRES):
        hist_val = getattr(optimizer_history, f"{key}_min")
        # replace by history if history has entry, or point does not match
        #  point recorded in result
        if hist_val is not None or not fval_match or not x_match:
            setattr(result, key, hist_val)

    return result


def read_result_from_file(
    problem: Problem,
    history_options: HistoryOptions,
    identifier: str,
) -> OptimizerResult:
    """Fill an OptimizerResult from history.

    Parameters
    ----------
    problem:
        The problem to find optimal parameters for.
    identifier:
        Multistart id.
    history_options:
        Optimizer history options.
    """
    if history_options.storage_file is None:
        raise ValueError("No history file specified.")

    if history_options.storage_file.endswith('.csv'):
        history = CsvHistory(
            file=history_options.storage_file.format(id=identifier),
            options=history_options,
            load_from_file=True,
        )
    elif history_options.storage_file.endswith(('.h5', '.hdf5')):
        history = Hdf5History.load(
            id=identifier,
            file=history_options.storage_file.format(id=identifier),
        )
    else:
        raise NotImplementedError()

    opt_hist = OptimizerHistory(
        history=history,
        x0=history.get_x_trace(0),
        lb=problem.lb,
        ub=problem.ub,
        generate_from_history=True,
    )

    result = OptimizerResult(
        id=identifier,
        message='loaded from file',
        exitflag=EXITFLAG_LOADED_FROM_FILE,
        time=max(history.get_time_trace()) if len(history) else 0.0,
    )
    result.id = identifier
    result = fill_result_from_history(
        result=result,
        optimizer_history=opt_hist,
    )
    result.update_to_full(problem)

    return result


def read_results_from_file(
    problem: Problem,
    history_options: HistoryOptions,
    n_starts: int,
) -> Result:
    """Fill a Result from a set of histories.

    Parameters
    ----------
    problem:
        The problem to find optimal parameters for.
    n_starts:
        Number of performed multistarts.
    history_options:
        Optimizer history options.
    """
    if history_options.storage_file is None:
        raise ValueError("No history file specified.")

    result = Result()
    result.problem = problem
    result.optimize_result = OptimizeResult()
    result.optimize_result.list = [
        read_result_from_file(problem, history_options, str(istart))
        for istart in range(n_starts)
        if os.path.exists(history_options.storage_file.format(id=str(istart)))
    ]
    if not result.optimize_result.list:
        logger.error("No history files found.")

    if len(result.optimize_result.list) != n_starts:
        logger.warning(
            f"History files were incomplete "
            f"({len(result.optimize_result.list)}/{n_starts})."
        )

    result.optimize_result.sort()
    return result


def optimization_result_from_history(
    filename: str,
    problem: pypesto.Problem,
) -> Result:
    """Convert a saved hdf5 History to an optimization result.

    Used for interrupted optimization runs.

    Parameters
    ----------
    filename:
        The name of the file in which the information are stored.
    problem:
        Problem, needed to identify what parameters to accept.

    Returns
    -------
        A result object in which the optimization result is constructed from
        history. But missing "Time, Message and Exitflag" keys.
    """
    result = Result()
    with h5py.File(filename, 'r') as f:
        for id_name in f['history'].keys():
            history = Hdf5History(id=id_name, file=filename)
            history.recover_options(filename)
            optimizer_history = OptimizerHistory(
                history=history,
                x0=f[f'history/{id_name}/trace/0/x'][()],
                lb=problem.lb,
                ub=problem.ub,
                generate_from_history=True,
            )
            optimizer_result = OptimizerResult(id=id_name)
            fill_result_from_history(optimizer_result, optimizer_history)
            result.optimize_result.append(optimizer_result)
    return result
