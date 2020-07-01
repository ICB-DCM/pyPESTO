"""
This is for testing the pypesto.Storage.
"""
import os
import tempfile

import numpy as np

from pypesto.store import (
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
