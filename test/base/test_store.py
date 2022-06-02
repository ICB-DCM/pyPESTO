"""Test the `pypesto.store` module."""

import os
import tempfile

import numpy as np
import scipy.optimize as so

import pypesto
import pypesto.optimize as optimize
import pypesto.profile as profile
import pypesto.sample as sample
from pypesto.C import (
    CHI2,
    EXITFLAG,
    FVAL,
    FVAL0,
    GRAD,
    HESS,
    ID,
    MESSAGE,
    N_FVAL,
    N_GRAD,
    N_HESS,
    N_RES,
    N_SRES,
    RES,
    SCHI2,
    SRES,
    X0,
    X,
)
from pypesto.optimize import optimization_result_from_history
from pypesto.store import (
    OptimizationResultHDF5Reader,
    OptimizationResultHDF5Writer,
    ProblemHDF5Reader,
    ProblemHDF5Writer,
    ProfileResultHDF5Reader,
    ProfileResultHDF5Writer,
    SamplingResultHDF5Reader,
    SamplingResultHDF5Writer,
    load_objective_config,
    read_result,
    write_result,
)

from ..visualize import (
    create_optimization_result,
    create_petab_problem,
    create_problem,
)


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
                        opt_res[key], read_result.optimize_result.list[i][key]
                    )
                else:
                    assert (
                        opt_res[key]
                        == read_result.optimize_result.list[i][key]
                    )


def test_storage_opt_result_update(hdf5_file):
    minimize_result = create_optimization_result()
    minimize_result_2 = create_optimization_result()

    result_file_name = hdf5_file

    opt_result_writer = OptimizationResultHDF5Writer(result_file_name)
    opt_result_writer.write(minimize_result)
    opt_result_writer.write(minimize_result_2, overwrite=True)
    opt_result_reader = OptimizationResultHDF5Reader(result_file_name)
    read_result = opt_result_reader.read()
    for i, opt_res in enumerate(minimize_result_2.optimize_result.list):
        for key in opt_res:
            if isinstance(opt_res[key], np.ndarray):
                np.testing.assert_array_equal(
                    opt_res[key], read_result.optimize_result.list[i][key]
                )
            else:
                assert opt_res[key] == read_result.optimize_result.list[i][key]


def test_storage_problem(hdf5_file):
    problem = create_problem()
    problem_writer = ProblemHDF5Writer(hdf5_file)
    problem_writer.write(problem)
    problem_reader = ProblemHDF5Reader(hdf5_file)
    read_problem = problem_reader.read()
    problem_attrs = [
        value
        for name, value in vars(ProblemHDF5Writer).items()
        if not name.startswith('_') and not callable(value)
    ]
    for attr in problem_attrs:
        if isinstance(problem.__dict__[attr], np.ndarray):
            np.testing.assert_array_equal(
                problem.__dict__[attr], read_problem.__dict__[attr]
            )
            assert isinstance(read_problem.__dict__[attr], np.ndarray)
        else:
            assert problem.__dict__[attr] == read_problem.__dict__[attr]
            assert not isinstance(read_problem.__dict__[attr], np.ndarray)


def test_storage_trace(hdf5_file):
    objective1 = pypesto.Objective(
        fun=so.rosen, grad=so.rosen_der, hess=so.rosen_hess
    )
    objective2 = pypesto.Objective(
        fun=so.rosen, grad=so.rosen_der, hess=so.rosen_hess
    )
    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))
    n_starts = 5
    startpoints = pypesto.startpoint.latin_hypercube(
        n_starts=n_starts, lb=lb, ub=ub
    )
    problem1 = pypesto.Problem(
        objective=objective1, lb=lb, ub=ub, x_guesses=startpoints
    )
    problem2 = pypesto.Problem(
        objective=objective2, lb=lb, ub=ub, x_guesses=startpoints
    )

    optimizer1 = optimize.ScipyOptimizer(options={'maxiter': 10})
    optimizer2 = optimize.ScipyOptimizer(options={'maxiter': 10})

    history_options_hdf5 = pypesto.HistoryOptions(
        trace_record=True, storage_file=hdf5_file
    )
    # optimize with history saved to hdf5
    result_hdf5 = optimize.minimize(
        problem=problem1,
        optimizer=optimizer1,
        n_starts=n_starts,
        history_options=history_options_hdf5,
        progress_bar=False,
    )

    # optimizing with history saved in memory
    history_options_memory = pypesto.HistoryOptions(trace_record=True)
    result_memory = optimize.minimize(
        problem=problem2,
        optimizer=optimizer2,
        n_starts=n_starts,
        history_options=history_options_memory,
        filename=None,
        progress_bar=False,
    )

    history_entries = [X, FVAL, GRAD, HESS, RES, SRES, CHI2, SCHI2]
    assert len(result_hdf5.optimize_result.list) == len(
        result_memory.optimize_result.list
    )
    for mem_res in result_memory.optimize_result.list:
        for hdf_res in result_hdf5.optimize_result.list:
            if mem_res['id'] == hdf_res['id']:
                for entry in history_entries:
                    hdf5_entry_trace = getattr(
                        hdf_res['history'], f'get_{entry}_trace'
                    )()
                    for iteration in range(len(hdf5_entry_trace)):
                        # comparing nan and None difficult
                        if (
                            hdf5_entry_trace[iteration] is None
                            or np.isnan(hdf5_entry_trace[iteration]).all()
                        ):
                            continue
                        np.testing.assert_array_equal(
                            getattr(
                                mem_res['history'], f'get_{entry}_trace'
                            )()[iteration],
                            hdf5_entry_trace[iteration],
                        )


def test_storage_profiling():
    """
    This test tests the saving and loading of profiles
    into HDF5 through pypesto.store.ProfileResultHDF5Writer
    and pypesto.store.ProfileResultHDF5Reader. Tests all entries
    aside from times and message.
    """
    objective = pypesto.Objective(
        fun=so.rosen, grad=so.rosen_der, hess=so.rosen_hess
    )
    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))
    n_starts = 5
    startpoints = pypesto.startpoint.latin_hypercube(
        n_starts=n_starts, lb=lb, ub=ub
    )
    problem = pypesto.Problem(
        objective=objective, lb=lb, ub=ub, x_guesses=startpoints
    )

    optimizer = optimize.ScipyOptimizer()

    result_optimization = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=n_starts,
        filename=None,
        progress_bar=False,
    )
    profile_original = profile.parameter_profile(
        problem=problem,
        result=result_optimization,
        profile_index=[0],
        optimizer=optimizer,
        filename=None,
        progress_bar=False,
    )

    fn = 'test_file.hdf5'
    try:
        pypesto_profile_writer = ProfileResultHDF5Writer(fn)
        pypesto_profile_writer.write(profile_original)
        pypesto_profile_reader = ProfileResultHDF5Reader(fn)
        profile_read = pypesto_profile_reader.read()

        for key in profile_original.profile_result.list[0][0].keys():
            if (
                profile_original.profile_result.list[0][0].keys is None
                or key == 'time_path'
            ):
                continue
            elif isinstance(
                profile_original.profile_result.list[0][0][key], np.ndarray
            ):
                np.testing.assert_array_equal(
                    profile_original.profile_result.list[0][0][key],
                    profile_read.profile_result.list[0][0][key],
                )
            elif isinstance(
                profile_original.profile_result.list[0][0][key], int
            ):
                assert (
                    profile_original.profile_result.list[0][0][key]
                    == profile_read.profile_result.list[0][0][key]
                )
    finally:
        if os.path.exists(fn):
            os.remove(fn)


def test_storage_sampling():
    """
    This test tests the saving and loading of samples
    into HDF5 through pypesto.store.SamplingResultHDF5Writer
    and pypesto.store.SamplingResultHDF5Reader. Tests all entries
    aside from time and message.
    """
    objective = pypesto.Objective(
        fun=so.rosen, grad=so.rosen_der, hess=so.rosen_hess
    )
    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))
    n_starts = 5
    startpoints = pypesto.startpoint.latin_hypercube(
        n_starts=n_starts, lb=lb, ub=ub
    )
    problem = pypesto.Problem(
        objective=objective, lb=lb, ub=ub, x_guesses=startpoints
    )

    optimizer = optimize.ScipyOptimizer()

    result_optimization = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=n_starts,
        filename=None,
        progress_bar=False,
    )
    x_0 = result_optimization.optimize_result.list[0]['x']
    sampler = sample.AdaptiveParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(),
        options={
            'show_progress': False,
        },
        n_chains=1,
    )
    sample_original = sample.sample(
        problem=problem,
        sampler=sampler,
        n_samples=100,
        x0=[x_0],
        filename=None,
    )

    fn = 'test_file.hdf5'
    try:
        pypesto_sample_writer = SamplingResultHDF5Writer(fn)
        pypesto_sample_writer.write(sample_original)
        pypesto_sample_reader = SamplingResultHDF5Reader(fn)
        sample_read = pypesto_sample_reader.read()

        for key in sample_original.sample_result.keys():
            if sample_original.sample_result[key] is None or key == 'time':
                continue
            elif isinstance(sample_original.sample_result[key], np.ndarray):
                np.testing.assert_array_equal(
                    sample_original.sample_result[key],
                    sample_read.sample_result[key],
                )
            elif isinstance(sample_original.sample_result[key], (float, int)):
                np.testing.assert_almost_equal(
                    sample_original.sample_result[key],
                    sample_read.sample_result[key],
                )
    finally:
        if os.path.exists(fn):
            os.remove(fn)


def test_storage_all():
    """Test `read_result` and `write_result`.

    It currently does not test read/write of the problem as this
    is know to not work completely. Also excludes testing the history
    key of an optimization result.
    """
    objective = pypesto.Objective(
        fun=so.rosen, grad=so.rosen_der, hess=so.rosen_hess
    )
    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))
    n_starts = 5
    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    optimizer = optimize.ScipyOptimizer()
    # Optimization
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=n_starts,
        filename=None,
        progress_bar=False,
    )
    # Profiling
    result = profile.parameter_profile(
        problem=problem,
        result=result,
        profile_index=[0],
        optimizer=optimizer,
        filename=None,
        progress_bar=False,
    )
    # Sampling
    sampler = sample.AdaptiveMetropolisSampler(
        options={
            'show_progress': False,
        }
    )
    result = sample.sample(
        problem=problem,
        sampler=sampler,
        n_samples=100,
        result=result,
        filename=None,
    )
    # Read and write
    filename = 'test_file.hdf5'
    try:
        write_result(result=result, filename=filename)
        result_read = read_result(filename=filename)

        # test optimize
        for i, opt_res in enumerate(result.optimize_result.list):
            for key in opt_res:
                if key == 'history':
                    continue
                if isinstance(opt_res[key], np.ndarray):
                    np.testing.assert_array_equal(
                        opt_res[key], result_read.optimize_result.list[i][key]
                    )
                else:
                    assert (
                        opt_res[key]
                        == result_read.optimize_result.list[i][key]
                    )

        # test profile
        for key in result.profile_result.list[0][0].keys():
            if (
                result.profile_result.list[0][0].keys is None
                or key == 'time_path'
            ):
                continue
            elif isinstance(result.profile_result.list[0][0][key], np.ndarray):
                np.testing.assert_array_equal(
                    result.profile_result.list[0][0][key],
                    result_read.profile_result.list[0][0][key],
                )
            elif isinstance(result.profile_result.list[0][0][key], int):
                assert (
                    result.profile_result.list[0][0][key]
                    == result_read.profile_result.list[0][0][key]
                )

        # test sample
        for key in result.sample_result.keys():
            if result.sample_result[key] is None or key == 'time':
                continue
            elif isinstance(result.sample_result[key], np.ndarray):
                np.testing.assert_array_equal(
                    result.sample_result[key],
                    result_read.sample_result[key],
                )
            elif isinstance(result.sample_result[key], (float, int)):
                np.testing.assert_almost_equal(
                    result.sample_result[key],
                    result_read.sample_result[key],
                )
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_storage_objective_config():
    """
    Test storing and loading the configuration
    of the objective function.
    """
    # create a problem with a function objective
    problem_fun = create_problem(2, x_names=['a', 'b'])
    # create a problem with amici Objective
    problem_amici = create_petab_problem()
    # put together into aggregated objective
    problem_agg = create_problem(2, x_names=['a', 'b'])
    objective_agg = pypesto.objective.AggregatedObjective(
        objectives=[problem_fun.objective, problem_amici.objective]
    )
    problem_agg.objective = objective_agg
    config_fun = problem_fun.objective.get_config()
    config_amici = problem_amici.objective.get_config()
    config_agg = objective_agg.get_config()

    fn = 'test_file.hdf5'
    try:
        writer = ProblemHDF5Writer(fn)
        writer.write(problem=problem_fun, overwrite=True)
        config_fun_r = load_objective_config(fn)
        writer.write(problem=problem_amici, overwrite=True)
        config_amici_r = load_objective_config(fn)
        writer.write(problem=problem_agg, overwrite=True)
        config_agg_r = load_objective_config(fn)

        # compare the different configurations
        for key in config_fun:
            assert config_fun[key] == config_fun_r[key]
        for key in config_amici:
            assert config_amici[key] == config_amici_r[key]
        for key in config_agg_r:
            if key == 'type':
                assert config_agg[key] == config_agg_r[key]
            else:
                assert len(config_agg_r[key]) == len(config_fun_r) or len(
                    config_agg_r[key]
                ) == len(config_amici_r)

    finally:
        if os.path.exists(fn):
            os.remove(fn)


def test_result_from_hdf5_history(hdf5_file):
    problem = create_petab_problem()

    history_options_hdf5 = pypesto.HistoryOptions(
        trace_record=True,
        storage_file=hdf5_file,
    )
    # optimize with history saved to hdf5
    result = optimize.minimize(
        problem=problem,
        n_starts=1,
        history_options=history_options_hdf5,
        progress_bar=False,
    )

    result_from_hdf5 = optimization_result_from_history(
        filename=hdf5_file, problem=problem
    )

    # Currently 'time' is not loaded.
    arguments = [
        ID,
        X,
        FVAL,
        GRAD,
        HESS,
        RES,
        SRES,
        N_FVAL,
        N_GRAD,
        N_HESS,
        N_RES,
        N_SRES,
        X0,
        FVAL0,
        EXITFLAG,
        MESSAGE,
    ]
    for key in arguments:
        if result.optimize_result.list[0][key] is None:
            assert result_from_hdf5.optimize_result.list[0][key] is None
        elif isinstance(result.optimize_result.list[0][key], np.ndarray):
            assert np.allclose(
                result.optimize_result.list[0][key],
                result_from_hdf5.optimize_result.list[0][key],
            ), key
        else:
            assert (
                result.optimize_result.list[0][key]
                == result_from_hdf5.optimize_result.list[0][key]
            ), key
