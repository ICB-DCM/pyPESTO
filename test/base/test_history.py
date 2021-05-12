"""
This is for testing the pypesto.History.
"""

import numpy as np
import pytest
import unittest
import tempfile
from typing import Sequence

import scipy.optimize as so
import pypesto
from pypesto.objective.util import sres_to_schi2, res_to_chi2
from pypesto import CsvHistory, HistoryOptions,\
    MemoryHistory, ObjectiveBase, Hdf5History
from pypesto.optimize.optimizer import read_result_from_file, OptimizerResult
from pypesto.objective.constants import (
    X, FVAL, GRAD, HESS, RES, SRES, CHI2, SCHI2)
from pypesto.engine import MultiProcessEngine

from ..util import rosen_for_sensi, load_amici_objective, CRProblem


class HistoryTest(unittest.TestCase):
    problem: pypesto.Problem = None
    optimizer: pypesto.optimize.Optimizer = None
    obj: ObjectiveBase = None
    history_options: HistoryOptions = None
    ub: np.ndarray = None
    lb: np.ndarray = None
    x_fixed_indices = None
    x_fixed_vals = None
    fix_pars = True

    def check_history(self):
        kwargs = {
            'objective': self.obj,
            'ub': self.ub,
            'lb': self.lb,
        }
        if self.fix_pars:
            kwargs = {**kwargs, **{
                'x_fixed_indices': self.x_fixed_indices,
                'x_fixed_vals': self.x_fixed_indices
            }}
        self.problem = pypesto.Problem(**kwargs)

        optimize_options = pypesto.optimize.OptimizeOptions(
            allow_failed_starts=False
        )

        self.history_options.trace_save_iter = 1

        for storage_file in ['tmp/traces/conversion_example_{id}.csv',
                             'tmp/traces/conversion_example_{id}.hdf5',
                             None]:
            self.history_options.storage_file = storage_file

            result = pypesto.optimize.minimize(
                problem=self.problem,
                optimizer=self.optimizer,
                n_starts=1,
                startpoint_method=pypesto.startpoint.uniform,
                options=optimize_options,
                history_options=self.history_options
            )

            for istart, start in enumerate(result.optimize_result.list):
                self.check_reconstruct_history(start, str(istart))
                self.check_load_from_file(start, str(istart))
                self.check_history_consistency(start)

    def check_load_from_file(self, start: OptimizerResult, id: str):
        """Verify we can reconstitute OptimizerResult from csv file"""

        if isinstance(start.history, MemoryHistory):
            return

        # TODO other implementations
        assert isinstance(start.history, (CsvHistory, Hdf5History))

        rstart = read_result_from_file(self.problem, self.history_options, id)

        result_attributes = [
            key for key in start.keys()
            if key not in ['history', 'message', 'exitflag', 'time']
        ]
        for attr in result_attributes:
            # if we didn't record we cant recover the value
            if not self.history_options.get(f'trace_record_{attr}', True):
                continue

            # note that we can expect slight deviations in grad when using
            # a ls optimizer since history computes this from res
            # with sensitivies activated while the optimizer uses a res
            # without sensitivities activated. If this fails to often,
            # increase atol
            if start[attr] is None:
                continue  # reconstituted may carry more information
            if attr in ['sres', 'grad', 'hess'] and rstart[attr] is None:
                continue  # may not always recover those
            elif isinstance(start[attr], np.ndarray):
                assert np.allclose(
                    start[attr], rstart[attr],
                    equal_nan=True, atol=1e-2,
                ), attr
            elif isinstance(start[attr], float):
                assert np.isclose(
                    start[attr], rstart[attr],
                    equal_nan=True
                ), attr
            else:
                assert start[attr] == rstart[attr], attr

    def check_reconstruct_history(self, start: OptimizerResult, id: str):
        """verify we can reconstruct history objects from csv/hdf5 files"""

        if isinstance(start.history, MemoryHistory):
            return

        assert isinstance(start.history, (CsvHistory, Hdf5History))

        if isinstance(start.history, CsvHistory):
            reconst_history = CsvHistory(
                file=self.history_options.storage_file.format(id=id),
                x_names=[self.problem.x_names[ix]
                         for ix in self.problem.x_free_indices],
                options=self.history_options,
                load_from_file=True
            )
        else:
            reconst_history = Hdf5History(
                file=self.history_options.storage_file.format(id=id),
                id=id,
                options=self.history_options
            )
        history_attributes = [
            a for a in dir(start.history)
            if not a.startswith('__')
            and not callable(getattr(start.history, a))
            and a not in ['options', '_abc_impl', '_start_time',
                          'start_time', '_trace', 'x_names']
        ]
        for attr in history_attributes:
            assert getattr(start.history, attr) == \
                   getattr(reconst_history, attr), attr

        assert len(start.history._trace) == len(reconst_history._trace)
        self.assertListEqual(start.history._trace.columns.to_list(),
                             reconst_history._trace.columns.to_list())
        for col in start.history._trace.columns:
            for true_val, reconst_val in zip(start.history._trace[col],
                                             reconst_history._trace[col]):
                if true_val is None:
                    assert reconst_val is None, col
                elif isinstance(true_val, float) and np.isnan(true_val):
                    assert np.isnan(reconst_val), col
                else:
                    assert np.isclose(true_val, reconst_val).all(), col

    def check_history_consistency(self, start: OptimizerResult):

        def xfull(x_trace):
            return self.problem.get_full_vector(
                x_trace, self.problem.x_fixed_vals
            )

        if isinstance(start.history, (CsvHistory, Hdf5History)):
            it_final = np.nanargmin(start.history.get_fval_trace())
            if isinstance(it_final, np.ndarray):
                it_final = it_final[0]
            it_start = int(np.where(np.logical_not(
                np.isnan(start.history.get_fval_trace())
            ))[0][0])
            assert np.allclose(
                xfull(start.history.get_x_trace(it_start)), start.x0)
            assert np.allclose(
                xfull(start.history.get_x_trace(it_final)), start.x)
            assert np.isclose(
                start.history.get_fval_trace(it_start), start.fval0)

        funs = {
            FVAL: self.obj.get_fval,
            GRAD: self.obj.get_grad,
            HESS: self.obj.get_hess,
            RES: self.obj.get_res,
            SRES: self.obj.get_sres,
            CHI2: lambda x: res_to_chi2(self.obj.get_res(x)),
            SCHI2: lambda x: sres_to_schi2(*self.obj(
                x, (0, 1,),
                pypesto.objective.constants.MODE_RES
            ))
        }
        for var, fun in funs.items():
            for it in range(5):
                x_full = xfull(start.history.get_x_trace(it))
                val = getattr(start.history, f'get_{var}_trace')(it)
                if not getattr(self.history_options, f'trace_record_{var}',
                               True):
                    assert np.isnan(val)
                    continue
                if np.all(np.isnan(val)):
                    continue
                if var in [FVAL, CHI2]:
                    # note that we can expect slight deviations here since
                    # this fval/chi2 may be computed without sensitivities
                    # while the result here may be computed with with
                    # sensitivies activated. If this fails to often,
                    # increase atol/rtol
                    assert np.isclose(
                        val, fun(x_full),
                        rtol=1e-3, atol=1e-4
                    ), var
                elif var in [RES]:
                    # note that we can expect slight deviations here since
                    # this res is computed without sensitivities while the
                    # result here may be computed with with sensitivies
                    # activated. If this fails to often, increase atol/rtol
                    assert np.allclose(
                        val, fun(x_full),
                        rtol=1e-3, atol=1e-4
                    ), var
                elif var in [SRES]:
                    assert np.allclose(
                        val, fun(x_full)[:, self.problem.x_free_indices],
                    ), var
                elif var in [GRAD, SCHI2]:
                    assert np.allclose(
                        val, self.problem.get_reduced_vector(fun(x_full)),
                    ), var
                elif var in [HESS]:
                    assert np.allclose(
                        val, self.problem.get_reduced_matrix(fun(x_full)),
                    ), var
                else:
                    raise RuntimeError('missing test implementation')


class ResModeHistoryTest(HistoryTest):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = pypesto.optimize.ScipyOptimizer(
            method='ls_trf',
            options={'max_nfev': 100}
        )
        cls.obj, _ = load_amici_objective(
            'conversion_reaction'
        )

        cls.lb = -2 * np.ones((1, 2))
        cls.ub = 2 * np.ones((1, 2))
        cls.x_fixed_indices = [0]
        cls.x_fixed_vals = [-0.3]

    def test_trace_chi2(self):
        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_chi2=True,
            trace_record_schi2=False,
        )

        self.check_history()

    def test_trace_chi2_schi2(self):
        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )

        self.check_history()

    def test_trace_schi2(self):
        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_chi2=True,
            trace_record_schi2=False,
        )

        self.check_history()

    def test_trace_grad(self):
        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
        )

        self.check_history()

    def test_trace_all(self):
        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=True,
            trace_record_res=True,
            trace_record_sres=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )

        self.fix_pars = False
        self.check_history()

    def test_trace_all_aggregated(self):
        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=True,
            trace_record_res=True,
            trace_record_sres=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )

        self.obj = pypesto.objective.AggregatedObjective([self.obj, self.obj])
        self.fix_pars = False
        self.check_history()


class CRResModeHistoryTest(HistoryTest):
    """Residual method test based on the conversion reaction model.

    This is useful to check that everything works also for a simple Objective,
    not only an AmiciObjective.
    """

    @classmethod
    def setUpClass(cls):
        cls.optimizer = pypesto.optimize.ScipyOptimizer(
            method='ls_trf',
            options={'max_nfev': 100}
        )
        problem = CRProblem()
        cls.obj = problem.get_objective(fim_for_hess=True)

        cls.lb = problem.lb
        cls.ub = problem.ub
        cls.x_fixed_indices = []
        cls.x_fixed_vals = []

    def test_trace_all(self):
        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=True,
            trace_record_res=True,
            trace_record_sres=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )

        self.fix_pars = False
        self.check_history()


class FunModeHistoryTest(HistoryTest):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = pypesto.optimize.ScipyOptimizer(
            method='trust-exact',
            options={'maxiter': 100}
        )

        cls.lb = 0 * np.ones((1, 2))
        cls.ub = 1 * np.ones((1, 2))
        cls.x_fixed_indices = [0]
        cls.x_fixed_vals = [0.0]

    def test_trace_grad(self):
        self.obj = rosen_for_sensi(
            max_sensi_order=2,
            integrated=False
        )['obj']

        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=False,
        )

        self.check_history()

    def test_trace_grad_integrated(self):
        self.obj = rosen_for_sensi(
            max_sensi_order=2,
            integrated=True
        )['obj']

        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=False,
        )

        self.check_history()

    def test_trace_all(self):
        self.obj = rosen_for_sensi(
            max_sensi_order=2,
            integrated=True
        )['obj']

        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=True,
            trace_record_res=True,
            trace_record_sres=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )
        self.fix_pars = False
        self.check_history()

    def test_trace_all_aggregated(self):
        self.obj = rosen_for_sensi(
            max_sensi_order=2,
            integrated=True
        )['obj']

        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=True,
            trace_record_res=True,
            trace_record_sres=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )
        self.obj = pypesto.objective.AggregatedObjective([self.obj, self.obj])
        self.fix_pars = False
        self.check_history()


class CRFunModeHistoryTest(HistoryTest):
    """Function method test based on the conversion reaction model.

    This is useful to check that everything works also for a simple Objective,
    not only an AmiciObjective.
    """

    @classmethod
    def setUpClass(cls):
        cls.optimizer = pypesto.optimize.ScipyOptimizer(
            method='trust-exact',
            options={'maxiter': 100}
        )
        problem = CRProblem()
        cls.obj = problem.get_objective(fim_for_hess=True)

        cls.lb = problem.lb
        cls.ub = problem.ub
        cls.x_fixed_indices = []
        cls.x_fixed_vals = []

    def test_trace_all(self):
        self.history_options = HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=True,
            trace_record_res=True,
            trace_record_sres=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )

        self.fix_pars = False
        self.check_history()


@pytest.fixture(params=["", "memory", "csv", "hdf5"])
def history(request) -> pypesto.History:
    if request.param == "memory":
        history = pypesto.MemoryHistory(options={'trace_record': True})
    elif request.param == "csv":
        file = tempfile.mkstemp(suffix='.csv')[1]
        history = pypesto.CsvHistory(file, options={'trace_record': True})
    else:
        history = pypesto.History()
    for _ in range(10):
        result = {FVAL: np.random.randn(), GRAD: np.random.randn(7)}
        history.update(np.random.randn(7), (0, 1), 'mode_fun', result)
    history.finalize()
    return history


def test_history_properties(history: pypesto.History):
    assert history.n_fval == 10
    assert history.n_grad == 10
    assert history.n_hess == 0
    assert history.n_res == 0
    assert history.n_sres == 0

    if type(history) == pypesto.History:
        with pytest.raises(NotImplementedError):
            history.get_fval_trace()
    else:
        fvals = history.get_fval_trace()
        assert len(fvals) == 10
        assert all(np.isfinite(fvals))

    if type(history) == pypesto.History:
        with pytest.raises(NotImplementedError):
            history.get_grad_trace()
    else:
        grads = history.get_grad_trace()
        assert len(grads) == 10
        assert len(grads[0]) == 7

    # assert x values are not all the same
    if type(history) != pypesto.History:
        xs = np.array(history.get_x_trace())
        assert np.all(xs[:-1] != xs[-1])

        ress = history.get_res_trace()
        assert all(np.isnan(res) for res in ress)


def test_trace_subset(history: pypesto.History):
    """Test whether selecting only a trace subset works."""
    if type(history) != pypesto.History:
        arr = list(range(0, len(history), 2))

        for var in ['fval', 'grad', 'hess', 'res', 'sres', 'chi2',
                    'schi2', 'x', 'time']:
            getter = getattr(history, f'get_{var}_trace')
            full_trace = getter()
            partial_trace = getter(arr)

            # check partial traces coincide
            assert len(partial_trace) == len(arr)
            for a, b in zip(partial_trace, [full_trace[i] for i in arr]):
                print(var, a, b)
                if var != 'schi2':
                    assert np.all(a == b) or np.isnan(a) and np.isnan(b)
                else:
                    assert np.all(a == b) or np.all(np.isnan(a)) \
                        and np.all(np.isnan(b))

            # check sequence type
            assert isinstance(full_trace, Sequence)
            assert isinstance(partial_trace, Sequence)

            # check individual type
            val = getter(0)
            if var in ['fval', 'chi2', 'time']:
                assert isinstance(val, float)
            else:
                assert isinstance(val, np.ndarray) or np.isnan(val)


def test_hdf5_history_mp():
    """Test whether hdf5-History works with a MultiProcessEngine."""
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

    optimizer1 = pypesto.optimize.ScipyOptimizer(options={'maxiter': 10})
    optimizer2 = pypesto.optimize.ScipyOptimizer(options={'maxiter': 10})

    with tempfile.TemporaryDirectory(dir=".") as tmpdirname:
        _, fn = tempfile.mkstemp(".hdf5", dir=f"{tmpdirname}")

        history_options_mp = pypesto.HistoryOptions(trace_record=True,
                                                    storage_file=fn)
        history_options_mem = pypesto.HistoryOptions(trace_record=True)
        # optimize with Memory History
        result_hdf5_mem = pypesto.optimize.minimize(
            problem=problem1, optimizer=optimizer1,
            n_starts=n_starts, history_options=history_options_mem,
            engine=MultiProcessEngine()
        )

        # optimizing with history saved in hdf5 and MultiProcessEngine
        result_memory_mp = pypesto.optimize.minimize(
            problem=problem2, optimizer=optimizer2,
            n_starts=n_starts, history_options=history_options_mp,
            engine=MultiProcessEngine()
        )

        history_entries = [X, FVAL, GRAD, HESS, RES, SRES, CHI2, SCHI2]
        assert len(result_hdf5_mem.optimize_result.list) == \
            len(result_memory_mp.optimize_result.list)
        for mp_res in result_memory_mp.optimize_result.list:
            for mem_res in result_hdf5_mem.optimize_result.list:
                if mp_res['id'] == mem_res['id']:
                    for entry in history_entries:
                        hdf5_entry_trace = getattr(mp_res['history'],
                                                   f'get_{entry}_trace')()
                        mem_entry_trace = getattr(mem_res['history'],
                                                  f'get_{entry}_trace')()
                        for iteration in range(len(hdf5_entry_trace)):
                            # comparing nan and None difficult
                            if hdf5_entry_trace[iteration] is None or np.isnan(
                                    hdf5_entry_trace[iteration]).all():
                                continue
                            np.testing.assert_array_equal(
                                mem_entry_trace[iteration],
                                hdf5_entry_trace[iteration])
