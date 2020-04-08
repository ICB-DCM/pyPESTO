"""
This is for testing the pypesto.Storage and history to HDF5.
"""
import tempfile
import numpy as np
import scipy as sp
from .visualize.test_visualize import create_problem, \
    create_optimization_result
from pypesto.storage import (
    ProblemHDF5Writer, ProblemHDF5Reader, OptimizationResultHDF5Writer,
    OptimizationResultHDF5Reader)
import pypesto


def test_storage_opt_result():
    minimize_result = create_optimization_result()
    with tempfile.TemporaryDirectory(dir=f".") as tmpdirname:
        _, fn = tempfile.mkstemp(".hdf5", dir=f"{tmpdirname}")
        opt_result_writer = OptimizationResultHDF5Writer(fn)
        opt_result_writer.write(minimize_result)
        opt_result_reader = OptimizationResultHDF5Reader(fn)
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


def test_storage_problem():
    problem = create_problem()
    with tempfile.TemporaryDirectory(dir=f".") as tmpdirname:
        _, fn = tempfile.mkstemp(".hdf5", dir=f"{tmpdirname}")
        problem_writer = ProblemHDF5Writer(fn)
        problem_writer.write(problem)
        problem_reader = ProblemHDF5Reader(fn)
        read_problem = problem_reader.read()
        problem_attrs = [value for name, value in
                         vars(ProblemHDF5Writer).items() if
                         not name.startswith('_') and not callable(value)]
        for attr in problem_attrs:
            if isinstance(read_problem.__dict__[attr], np.ndarray):
                np.testing.assert_array_equal(
                    problem.__dict__[attr],
                    read_problem.__dict__[attr])
            else:
                assert problem.__dict__[attr] == \
                       read_problem.__dict__[attr]


def test_storage_trace():
    objective1 = pypesto.Objective(fun=sp.optimize.rosen,
                                   grad=sp.optimize.rosen_der,
                                   hess=sp.optimize.rosen_hess)
    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))
    problem1 = pypesto.Problem(objective=objective1, lb=lb, ub=ub)
    optimizer_bfgs = pypesto.ScipyOptimizer(method='l-bfgs-b')
    n_starts = 20

    with tempfile.TemporaryDirectory(dir=f".") as tmpdirname:
        _, fn = tempfile.mkstemp(".hdf5", dir=f"{tmpdirname}")

        history_options = pypesto.HistoryOptions(trace_record=True,
                                                 storage_file=fn)
        result1_bfgs = pypesto.minimize(
            problem=problem1, optimizer=optimizer_bfgs,
            n_starts=n_starts, history_options=history_options)
