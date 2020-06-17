"""
This is for testing the pypesto.Storage and history to HDF5.
"""
import os
import tempfile
import pytest
import numpy as np
import scipy as sp
import scipy.optimize as so
import pypesto
from pypesto.objective.constants import (X, FVAL, GRAD, HESS, RES, SRES, CHI2,
                                         SCHI2, TIME)

from pypesto.storage import (
    ProblemHDF5Writer, ProblemHDF5Reader, OptimizationResultHDF5Writer,
    OptimizationResultHDF5Reader)
from .visualize.test_visualize import create_problem, \
    create_optimization_result


def test_storage_opt_result():
    minimize_result = create_optimization_result()
    with tempfile.TemporaryDirectory(dir=".") as tmpdirname:
        result_file_name = os.path.join(tmpdirname, "a", "b", "result.h5")
        opt_result_writer = OptimizationResultHDF5Writer(result_file_name)
        opt_result_writer.write(minimize_result)
        opt_result_reader = OptimizationResultHDF5Reader(result_file_name)
        read_result = opt_result_reader.read()
        for i, opt_res in enumerate(minimize_result.optimize_result.list):
            for key in opt_res:
                if isinstance(opt_res[key], np.ndarray):
                    np.testing.assert_array_equal(
                        opt_res[key],
                        read_result.optimize_result.list[i][key])
                else:
                    assert opt_res[key] == \
                           read_result.optimize_result.list[i][key]


def test_storage_opt_result_update():
    minimize_result = create_optimization_result()
    minimize_result_2 = create_optimization_result()
    with tempfile.TemporaryDirectory(dir=".") as tmpdirname:
        result_file_name = os.path.join(tmpdirname, "a", "b", "result.h5")
        opt_result_writer = OptimizationResultHDF5Writer(result_file_name)
        opt_result_writer.write(minimize_result)
        opt_result_writer.write(minimize_result_2, overwrite=True)
        opt_result_reader = OptimizationResultHDF5Reader(result_file_name)
        read_result = opt_result_reader.read()
        for i, opt_res in enumerate(minimize_result_2.optimize_result.list):
            for key in opt_res:
                if isinstance(opt_res[key], np.ndarray):
                    np.testing.assert_array_equal(
                        opt_res[key],
                        read_result.optimize_result.list[i][key])
                else:
                    assert opt_res[key] == \
                           read_result.optimize_result.list[i][key]


def test_storage_problem():
    problem = create_problem()
    with tempfile.TemporaryDirectory(dir=".") as tmpdirname:
        _, fn = tempfile.mkstemp(".hdf5", dir=f"{tmpdirname}")
        problem_writer = ProblemHDF5Writer(fn)
        problem_writer.write(problem)
        problem_reader = ProblemHDF5Reader(fn)
        read_problem = problem_reader.read()
        problem_attrs = [value for name, value in
                         vars(ProblemHDF5Writer).items() if
                         not name.startswith('_') and not callable(value)]
        for attr in problem_attrs:
            if isinstance(problem.__dict__[attr], np.ndarray):
                np.testing.assert_array_equal(
                    problem.__dict__[attr],
                    read_problem.__dict__[attr])
                assert isinstance(read_problem.__dict__[attr], np.ndarray)
            else:
                assert problem.__dict__[attr] == \
                       read_problem.__dict__[attr]
                assert not isinstance(read_problem.__dict__[attr], np.ndarray)


@pytest.mark.skip(reason="strange error in memory history")
def test_storage_trace():
    objective1 = pypesto.Objective(fun=so.rosen,
                                   grad=so.rosen_der,
                                   hess=so.rosen_hess)
    objective2 = pypesto.Objective(fun=so.rosen,
                                   grad=so.rosen_der,
                                   hess=so.rosen_hess)
    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))
    n_starts = 20
    startpoints = pypesto.startpoint.latin_hypercube(n_starts=n_starts,
                                                     lb=lb,
                                                     ub=ub)
    problem1 = pypesto.Problem(objective=objective1, lb=lb, ub=ub)
    problem2 = pypesto.Problem(objective=objective2, lb=lb, ub=ub)
    problem1.x_guesses = startpoints
    problem2.x_guesses = startpoints
    optimizer1 = pypesto.ScipyOptimizer()
    optimizer2 = pypesto.ScipyOptimizer()

    with tempfile.TemporaryDirectory(dir=f".") as tmpdirname:
        _, fn = tempfile.mkstemp(".hdf5", dir=f"{tmpdirname}")

        history_options_hdf5 = pypesto.HistoryOptions(trace_record=True,
                                                      storage_file="test.hdf5")
        # optimize with history saved to hdf5
        result_hdf5 = pypesto.minimize(
            problem=problem1, optimizer=optimizer1,
            n_starts=n_starts, history_options=history_options_hdf5)

        # optimizing with history saved in memory
        history_options_memory = pypesto.HistoryOptions(trace_record=True)
        result_memory = pypesto.minimize(
            problem=problem2, optimizer=optimizer2,
            n_starts=n_starts, history_options=history_options_memory)

        history_entries = [X, FVAL, GRAD, HESS, RES, SRES, CHI2, SCHI2, TIME]
        assert len(result_memory.optimize_result.list) == \
               len(result_memory.optimize_result.list)
        for mem_res in result_memory.optimize_result.list:
            for hdf_res in result_hdf5.optimize_result.list:
                if mem_res['id'] == hdf_res['id']:
                    for entry in history_entries:
                        np.testing.assert_array_equal(getattr(mem_res[
                                                                  'history'],
                            f'get_{entry}_trace')(), getattr(
                            hdf_res['history'], f'get_{entry}_trace')())
