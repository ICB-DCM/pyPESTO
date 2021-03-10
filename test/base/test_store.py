"""
This is for testing the pypesto.Storage.
"""
import os
import tempfile
import pypesto
import pypesto.profile as profile

from pypesto.objective.constants import (X, FVAL, GRAD,
                                         HESS, RES, SRES,
                                         CHI2, SCHI2)
import scipy.optimize as so

import numpy as np

from pypesto.store import (
    ProblemHDF5Writer, ProblemHDF5Reader, OptimizationResultHDF5Writer,
    OptimizationResultHDF5Reader, ProfileResultHDF5Writer,
    ProfileResultHDF5Reader)
from ..visualize import create_problem, create_optimization_result


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
    n_starts = 5
    startpoints = pypesto.startpoint.latin_hypercube(n_starts=n_starts,
                                                     lb=lb,
                                                     ub=ub)
    problem1 = pypesto.Problem(objective=objective1, lb=lb, ub=ub,
                               x_guesses=startpoints)
    problem2 = pypesto.Problem(objective=objective2, lb=lb, ub=ub,
                               x_guesses=startpoints)

    optimizer1 = pypesto.optimize.ScipyOptimizer(options={'maxiter': 100})
    optimizer2 = pypesto.optimize.ScipyOptimizer(options={'maxiter': 100})

    with tempfile.TemporaryDirectory(dir=".") as tmpdirname:
        _, fn = tempfile.mkstemp(".hdf5", dir=f"{tmpdirname}")

        history_options_hdf5 = pypesto.HistoryOptions(trace_record=True,
                                                      storage_file=fn)
        # optimize with history saved to hdf5
        result_hdf5 = pypesto.optimize.minimize(
            problem=problem1, optimizer=optimizer1,
            n_starts=n_starts, history_options=history_options_hdf5)

        # optimizing with history saved in memory
        history_options_memory = pypesto.HistoryOptions(trace_record=True)
        result_memory = pypesto.optimize.minimize(
            problem=problem2, optimizer=optimizer2,
            n_starts=n_starts, history_options=history_options_memory)

        history_entries = [X, FVAL, GRAD, HESS, RES, SRES, CHI2, SCHI2]
        assert len(result_hdf5.optimize_result.list) == \
            len(result_memory.optimize_result.list)
        for mem_res in result_memory.optimize_result.list:
            for hdf_res in result_hdf5.optimize_result.list:
                if mem_res['id'] == hdf_res['id']:
                    for entry in history_entries:
                        hdf5_entry_trace = getattr(hdf_res['history'],
                                                   f'get_{entry}_trace')()
                        for iteration in range(len(hdf5_entry_trace)):
                            print(hdf5_entry_trace[iteration])
                            # comparing nan and None difficult
                            if hdf5_entry_trace[iteration] is None or np.isnan(
                                    hdf5_entry_trace[iteration]).all():
                                continue
                            print(getattr(mem_res['history'],
                                          f'get_{entry}_trace')()[iteration])
                            print(entry, iteration)
                            np.testing.assert_array_equal(
                                getattr(mem_res['history'],
                                        f'get_{entry}_trace')()[iteration],
                                hdf5_entry_trace[iteration])


def test_storage_profiling():
    objective = pypesto.Objective(fun=so.rosen,
                                  grad=so.rosen_der,
                                  hess=so.rosen_hess)
    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))
    n_starts = 5
    startpoints = pypesto.startpoint.latin_hypercube(n_starts=n_starts,
                                                     lb=lb,
                                                     ub=ub)
    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub,
                              x_guesses=startpoints)

    optimizer = pypesto.optimize.ScipyOptimizer()

    result_optimization = pypesto.optimize.minimize(
        problem=problem, optimizer=optimizer,
        n_starts=n_starts)
    profile_original = profile.parameter_profile(
        problem=problem, result=result_optimization,
        profile_index=[0], optimizer=optimizer)
    with tempfile.TemporaryDirectory(dir=".") as tmpdirname:
        _, fn = tempfile.mkstemp(".hdf5", dir=f"{tmpdirname}")

    pypesto_profile_writer = ProfileResultHDF5Writer(fn)
    pypesto_profile_writer.write(profile_original)
    pypesto_profile_reader = ProfileResultHDF5Reader(fn)
    profile_read = pypesto_profile_reader.read()

    # compare the x_paths of both profiles
    np.testing.assert_array_equal(
        profile_original.profile_result.list[0][0]['x_path'],
        profile_read.profile_result.list[0][0]['x_path']
    )
