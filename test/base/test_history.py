"""Test the :class:`pypesto.History`."""

import tempfile
import unittest
from typing import Sequence

import numpy as np
import pytest
import scipy.optimize as so

import pypesto
import pypesto.optimize as optimize
from pypesto import (
    CsvHistory,
    Hdf5History,
    HistoryOptions,
    MemoryHistory,
    ObjectiveBase,
)
from pypesto.C import CHI2, FVAL, GRAD, HESS, RES, SCHI2, SRES, X
from pypesto.engine import MultiProcessEngine
from pypesto.objective.util import res_to_chi2, sres_to_schi2

from ..util import CRProblem, load_amici_objective, rosen_for_sensi


class HistoryTest(unittest.TestCase):
    problem: pypesto.Problem = None
    optimizer: optimize.Optimizer = None
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
            kwargs = {
                **kwargs,
                **{
                    'x_fixed_indices': self.x_fixed_indices,
                    'x_fixed_vals': self.x_fixed_indices,
                },
            }
        self.problem = pypesto.Problem(**kwargs)

        optimize_options = pypesto.optimize.OptimizeOptions(
            allow_failed_starts=False,
            history_beats_optimizer=True,
        )

        self.history_options.trace_save_iter = 1

        for storage_type in ['.csv', '.hdf5', None]:
            with tempfile.TemporaryDirectory(dir=".") as tmpdir:
                if storage_type == ".csv":
                    _, fn = tempfile.mkstemp(
                        "_{id}" + storage_type, dir=tmpdir
                    )
                else:
                    _, fn = tempfile.mkstemp(storage_type, dir=tmpdir)
                self.history_options.storage_file = fn
                if storage_type is None:
                    self.history_options.storage_file = None

                n_starts = 1

                result = pypesto.optimize.minimize(
                    problem=self.problem,
                    optimizer=self.optimizer,
                    n_starts=n_starts,
                    startpoint_method=pypesto.startpoint.uniform,
                    options=optimize_options,
                    history_options=self.history_options,
                    filename=None,
                    progress_bar=False,
                )

                for istart, start in enumerate(result.optimize_result.list):
                    self.check_reconstruct_history(start, str(istart))
                    self.check_load_from_file(start, str(istart))
                    self.check_history_consistency(start)

                # check that we can also aggregate from multiple files.
                # load more results than what is generated to check whether
                # this also works in case of incomplete results.
                if storage_type is not None:
                    optimize.read_results_from_file(
                        self.problem,
                        self.history_options,
                        n_starts=n_starts + 1,
                    )
                else:
                    with pytest.raises(ValueError):
                        optimize.read_results_from_file(
                            self.problem,
                            self.history_options,
                            n_starts=n_starts,
                        )

    def check_load_from_file(self, start: pypesto.OptimizerResult, id: str):
        """Verify we can reconstitute OptimizerResult from history file"""
        # TODO other implementations
        if isinstance(start.history, MemoryHistory):
            return
        assert isinstance(start.history, (CsvHistory, Hdf5History))

        rstart = optimize.read_result_from_file(
            self.problem, self.history_options, id
        )

        result_attributes = [
            key
            for key in start.keys()
            if key
            not in ['history', 'message', 'exitflag', 'time', 'optimizer']
        ]
        for attr in result_attributes:
            # if we didn't record we can't recover the value
            if not self.history_options.get(f'trace_record_{attr}', True):
                continue

            # note that we can expect slight deviations in grad when using
            # a ls optimizer since history computes this from res
            # with sensitivities activated while the optimizer uses a res
            # without sensitivities activated. If this fails too often,
            # increase atol
            if start[attr] is None:
                continue  # reconstituted may carry more information
            if attr in ['sres', 'grad', 'hess'] and rstart[attr] is None:
                continue  # may not always recover those
            elif isinstance(start[attr], np.ndarray):
                assert np.allclose(
                    start[attr],
                    rstart[attr],
                    equal_nan=True,
                    atol=1e-2,
                ), attr
            elif isinstance(start[attr], float):
                assert np.isclose(
                    start[attr], rstart[attr], equal_nan=True
                ), attr
            else:
                assert start[attr] == rstart[attr], attr

    def check_reconstruct_history(
        self, start: pypesto.OptimizerResult, id: str
    ):
        """verify we can reconstruct history objects from csv/hdf5 files"""

        if isinstance(start.history, MemoryHistory):
            return

        assert isinstance(start.history, (CsvHistory, Hdf5History))

        if isinstance(start.history, CsvHistory):
            reconst_history = CsvHistory(
                file=self.history_options.storage_file.format(id=id),
                x_names=[
                    self.problem.x_names[ix]
                    for ix in self.problem.x_free_indices
                ],
                options=self.history_options,
                load_from_file=True,
            )
        else:
            reconst_history = Hdf5History(
                file=self.history_options.storage_file.format(id=id),
                id=id,
                options=self.history_options,
            )
        history_attributes = [
            a
            for a in dir(start.history)
            if not a.startswith('__')
            and not callable(getattr(start.history, a))
            and a
            not in [
                'options',
                '_abc_impl',
                '_start_time',
                'start_time',
                '_trace',
                'x_names',
                'editable',
            ]
        ]
        for attr in history_attributes:
            assert getattr(start.history, attr) == getattr(
                reconst_history, attr
            ), attr

        assert len(start.history) == len(reconst_history)

        history_entries = [X, FVAL, GRAD, HESS, RES, SRES, CHI2, SCHI2]

        for entry in history_entries:
            original_trace = getattr(start.history, f'get_{entry}_trace')()
            reconst_trace = getattr(reconst_history, f'get_{entry}_trace')()
            for iteration in range(len(original_trace)):
                # comparing nan and None difficult
                if original_trace[iteration] is None:
                    assert reconst_trace[iteration] is None
                if np.isnan(original_trace[iteration]).all():
                    assert np.isnan(reconst_trace[iteration]).all()
                np.testing.assert_array_almost_equal(
                    reconst_trace[iteration],
                    original_trace[iteration],
                    decimal=10,
                )

    def check_history_consistency(self, start: pypesto.OptimizerResult):
        def xfull(x_trace):
            return self.problem.get_full_vector(
                x_trace, self.problem.x_fixed_vals
            )

        if isinstance(start.history, (CsvHistory, Hdf5History)):
            # get index of optimal parameter
            ix_admit = [
                ix
                for ix, x in enumerate(start.history.get_x_trace())
                if np.all(x >= self.problem.lb)
                and np.all(x <= self.problem.ub)
            ]
            it_final = np.nanargmin(start.history.get_fval_trace(ix_admit))
            if isinstance(it_final, np.ndarray):
                it_final = it_final[0]
            it_final = ix_admit[it_final]

            it_start = int(
                np.where(
                    np.logical_not(np.isnan(start.history.get_fval_trace()))
                )[0][0]
            )
            assert np.allclose(
                xfull(start.history.get_x_trace(it_start)), start.x0
            ), type(start.history)
            assert np.allclose(
                xfull(start.history.get_x_trace(it_final)), start.x
            ), type(start.history)
            assert np.isclose(
                start.history.get_fval_trace(it_start), start.fval0
            ), type(start.history)

        funs = {
            FVAL: self.obj.get_fval,
            GRAD: self.obj.get_grad,
            HESS: self.obj.get_hess,
            RES: self.obj.get_res,
            SRES: self.obj.get_sres,
            CHI2: lambda x: res_to_chi2(self.obj.get_res(x)),
            SCHI2: lambda x: sres_to_schi2(
                *self.obj(
                    x,
                    (
                        0,
                        1,
                    ),
                    pypesto.C.MODE_RES,
                )
            ),
        }
        for var, fun in funs.items():
            for it in range(5):
                x_full = xfull(start.history.get_x_trace(it))
                val = getattr(start.history, f'get_{var}_trace')(it)

                if not getattr(
                    self.history_options, f'trace_record_{var}', True
                ):
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
                        val, fun(x_full), rtol=1e-3, atol=1e-4
                    ), var
                elif var in [RES]:
                    # note that we can expect slight deviations here since
                    # this res is computed without sensitivities while the
                    # result here may be computed with with sensitivies
                    # activated. If this fails too often, increase atol/rtol
                    assert np.allclose(
                        val, fun(x_full), rtol=1e-3, atol=1e-4
                    ), var
                elif var in [SRES]:
                    assert np.allclose(
                        val,
                        fun(x_full)[:, self.problem.x_free_indices],
                    ), var
                elif var in [GRAD, SCHI2]:
                    assert np.allclose(
                        val,
                        self.problem.get_reduced_vector(fun(x_full)),
                    ), var
                elif var in [HESS]:
                    assert np.allclose(
                        val,
                        self.problem.get_reduced_matrix(fun(x_full)),
                    ), var
                else:
                    raise RuntimeError('missing test implementation')


class ResModeHistoryTest(HistoryTest):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = pypesto.optimize.ScipyOptimizer(
            method='ls_trf', options={'max_nfev': 100}
        )
        cls.obj, _ = load_amici_objective('conversion_reaction')

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
            trace_record_chi2=False,
            trace_record_schi2=True,
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
            method='ls_trf', options={'max_nfev': 100}
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
            options={'maxiter': 100},
        )

        cls.lb = 0 * np.ones((1, 2))
        cls.ub = 1 * np.ones((1, 2))
        cls.x_fixed_indices = [0]
        cls.x_fixed_vals = [0.0]

    def test_trace_grad(self):
        self.obj = rosen_for_sensi(
            max_sensi_order=2,
            integrated=False,
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
            integrated=True,
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
            integrated=True,
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
        self.obj = rosen_for_sensi(max_sensi_order=2, integrated=True)['obj']

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
            method='trust-exact', options={'maxiter': 100}
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

        for var in [
            'fval',
            'grad',
            'hess',
            'res',
            'sres',
            'chi2',
            'schi2',
            'x',
            'time',
        ]:
            getter = getattr(history, f'get_{var}_trace')
            full_trace = getter()
            partial_trace = getter(arr)

            # check partial traces coincide
            assert len(partial_trace) == len(arr)
            for a, b in zip(partial_trace, [full_trace[i] for i in arr]):
                if var != 'schi2':
                    assert np.all(a == b) or np.isnan(a) and np.isnan(b)
                else:
                    assert (
                        np.all(a == b)
                        or np.all(np.isnan(a))
                        and np.all(np.isnan(b))
                    )

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

    optimizer1 = pypesto.optimize.ScipyOptimizer(options={'maxiter': 10})
    optimizer2 = pypesto.optimize.ScipyOptimizer(options={'maxiter': 10})

    with tempfile.TemporaryDirectory(dir=".") as tmpdirname:
        _, fn = tempfile.mkstemp(".hdf5", dir=f"{tmpdirname}")

        history_options_mp = pypesto.HistoryOptions(
            trace_record=True, storage_file=fn
        )
        history_options_mem = pypesto.HistoryOptions(trace_record=True)
        # optimize with Memory History
        result_hdf5_mem = pypesto.optimize.minimize(
            problem=problem1,
            optimizer=optimizer1,
            n_starts=n_starts,
            history_options=history_options_mem,
            engine=MultiProcessEngine(),
            filename=None,
            progress_bar=False,
        )

        # optimizing with history saved in hdf5 and MultiProcessEngine
        result_memory_mp = pypesto.optimize.minimize(
            problem=problem2,
            optimizer=optimizer2,
            n_starts=n_starts,
            history_options=history_options_mp,
            engine=MultiProcessEngine(),
            filename=None,
            progress_bar=False,
        )

        history_entries = [X, FVAL, GRAD, HESS, RES, SRES, CHI2, SCHI2]
        assert len(result_hdf5_mem.optimize_result.list) == len(
            result_memory_mp.optimize_result.list
        )
        for mp_res in result_memory_mp.optimize_result.list:
            for mem_res in result_hdf5_mem.optimize_result.list:
                if mp_res['id'] == mem_res['id']:
                    for entry in history_entries:
                        hdf5_entry_trace = getattr(
                            mp_res['history'], f'get_{entry}_trace'
                        )()
                        mem_entry_trace = getattr(
                            mem_res['history'], f'get_{entry}_trace'
                        )()
                        for iteration in range(len(hdf5_entry_trace)):
                            # comparing nan and None difficult
                            if (
                                hdf5_entry_trace[iteration] is None
                                or np.isnan(hdf5_entry_trace[iteration]).all()
                            ):
                                continue
                            np.testing.assert_array_equal(
                                mem_entry_trace[iteration],
                                hdf5_entry_trace[iteration],
                            )


def test_trim_history():
    """
    Test whether the history gets correctly trimmed to be monotonically
    decreasing.
    """
    problem = CRProblem()
    pypesto_problem = problem.get_problem()

    optimizer = pypesto.optimize.ScipyOptimizer()
    history_options = pypesto.HistoryOptions(trace_record=True)
    result = pypesto.optimize.minimize(
        problem=pypesto_problem,
        optimizer=optimizer,
        n_starts=1,
        history_options=history_options,
        filename=None,
        progress_bar=False,
    )
    fval_trace = result.optimize_result.list[0].history.get_fval_trace()
    fval_trace_trimmed = result.optimize_result.list[0].history.get_fval_trace(
        trim=True
    )
    fval_trimmed_man = []
    fval_current = np.inf
    for fval_i in fval_trace:
        if fval_i <= fval_current:
            fval_trimmed_man.append(fval_i)
            fval_current = fval_i
    assert fval_trace_trimmed == fval_trimmed_man
