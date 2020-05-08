"""
This is for testing the pypesto.History.
"""

import numpy as np
import pytest
import pypesto
from pypesto.objective.constants import FVAL, GRAD
import unittest
import tempfile

from test.test_objective import rosen_for_sensi
from test.test_sbml_conversion import load_model_objective
from pypesto.objective.util import sres_to_schi2, res_to_chi2
from pypesto.objective import CsvHistory
from pypesto.optimize.optimizer import read_result_from_csv


class HistoryTest(unittest.TestCase):
    problem: pypesto.Problem = None
    optimizer: pypesto.Optimizer = None
    obj: pypesto.Objective = None
    ub: np.ndarray = None
    lb: np.ndarray = None
    x_fixed_indices = None
    x_fixed_vals = None

    def check_history(self):
        self.problem = pypesto.Problem(self.obj, self.lb, self.ub,
                                       x_fixed_indices=self.x_fixed_indices,
                                       x_fixed_vals=self.x_fixed_indices)

        def xfull(x_trace):
            return self.problem.get_full_vector(
                x_trace, self.problem.x_fixed_vals
            )

        optimize_options = pypesto.OptimizeOptions(
            allow_failed_starts=False
        )

        storage_file = 'tmp/traces/conversion_example_{id}.csv'
        history_options = pypesto.HistoryOptions(
            trace_record=True,
            trace_record_hess=False,
            trace_save_iter=1,
            storage_file=storage_file,
        )

        result = pypesto.minimize(
            problem=self.problem,
            optimizer=self.optimizer,
            n_starts=1,
            startpoint_method=pypesto.startpoint.uniform,
            options=optimize_options,
            history_options=history_options
        )

        # disable trace from here on
        self.obj.history.options.trace_record = False
        for istart, start in enumerate(result.optimize_result.list):

            # verify we can reconstruct history objects from csv files
            reconst_history = CsvHistory(
                file=storage_file.format(id=str(istart)),
                options=history_options,
                load_from_file=True
            )
            history_attributes = [
                a for a in dir(start.history)
                if not a.startswith('__')
                and not callable(getattr(start.history, a))
                and a not in ['options', '_abc_impl', '_start_time',
                              'start_time', '_trace']
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

            # verify we can reconstitute OptimizerResult from csv file
            rstart = read_result_from_csv(self.problem, history_options,
                                          str(istart))

            result_attributes = [
                key for key in start.keys()
                if key not in ['history', 'message', 'exitflag', 'time']
            ]
            for attr in result_attributes:
                record_bool = f'trace_record_{attr}'
                # if we didn't record we cant recover the value
                if record_bool in dir(history_options):
                    if not getattr(history_options, record_bool):
                        continue

                # note that we can expect slight deviations in grad when using
                # a ls optimizer since history computes this from res
                # with sensitivies activated while the optimizer uses a res
                # without sensitivities activated. If this fails to often,
                # increase atol
                if start[attr] is None:
                    continue  # reconstituted may carry more information
                elif isinstance(start[attr], np.ndarray):
                    assert np.allclose(
                        start[attr], rstart[attr],
                        equal_nan=True, atol=1e-3
                    ), attr
                elif isinstance(start[attr], float):
                    assert np.isclose(
                        start[attr], rstart[attr],
                        equal_nan=True
                    ), attr
                else:
                    assert start[attr] == rstart[attr], attr

            # verify consistency of stored values
            trace = start.history._trace

            it_final = int(trace[('fval', np.NaN)].idxmin())
            it_start = int(np.where(np.logical_not(
                np.isnan(trace['fval'].values)
            ))[0][0])
            assert np.allclose(xfull(trace['x'].values[0, :]), start.x0)
            assert np.allclose(xfull(trace['x'].values[it_final, :]), start.x)
            assert np.isclose(trace['fval'].values[it_start, 0], start.fval0)

            funs = {
                'fval': self.obj.get_fval,
                'grad': self.obj.get_grad,
                'hess': self.obj.get_hess,
                'res': self.obj.get_res,
                'sres': self.obj.get_sres,
                'chi2': lambda x: res_to_chi2(self.obj.get_res(x)),
                'schi2': lambda x: sres_to_schi2(*self.obj(
                    x,
                    (0, 1,),
                    pypesto.objective.constants.MODE_RES
                ))
            }
            for var, fun in funs.items():
                for it in range(5):
                    x_full = xfull(trace['x'].values[it, :])
                    if var in ['fval', 'chi2']:
                        if not np.isnan(trace[var].values[it, 0]):
                            assert np.isclose(
                                trace[var].values[it, 0], fun(x_full)
                            ), var
                    elif var in ['res']:
                        if trace[var].values[it, 0] is not None:
                            assert np.allclose(
                                trace[var].values[it, 0], fun(x_full)
                            ), var
                    elif var in ['sres']:
                        if trace[var].values[it, 0] is not None:
                            assert np.allclose(
                                trace[var].values[it, 0],
                                fun(x_full)[:, self.problem.x_free_indices]
                            ), var
                    elif var in ['grad', 'schi2']:
                        if not np.isnan(trace[var].values[it, :]).all():
                            assert np.allclose(
                                trace[var].values[it, :],
                                self.problem.get_reduced_vector(fun(x_full))
                            ), var
                    elif var in ['hess']:
                        if not trace[var].values[it, 0] is None and \
                                not np.isnan(trace[var].values[it, :]).all():
                            assert np.allclose(
                                trace[var].values[it, :],
                                self.problem.get_reduced_matrix(fun(x_full))
                            ), var
                    else:
                        raise RuntimeError('missing test implementation')


class ResModeHistoryTest(HistoryTest):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = pypesto.ScipyOptimizer(
            method='ls_trf',
            options={'max_nfev': 100}
        )
        cls.obj, _ = load_model_objective(
            'conversion_reaction'
        )

        cls.lb = -2 * np.ones((1, 2))
        cls.ub = 2 * np.ones((1, 2))
        cls.x_fixed_indices = [0]
        cls.x_fixed_vals = [-0.3]

    def test_trace_chi2(self):
        self.obj.history.options = pypesto.HistoryOptions(
            trace_record=True,
            trace_record_chi2=True,
            trace_record_schi2=False,
        )

        self.check_history()

    def test_trace_chi2_schi2(self):
        self.obj.history.options = pypesto.HistoryOptions(
            trace_record=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )

        self.check_history()

    def test_trace_schi2(self):
        self.obj.history.options = pypesto.HistoryOptions(
            trace_record=True,
            trace_record_chi2=True,
            trace_record_schi2=False,
        )

        self.check_history()

    def test_trace_grad(self):
        self.obj.history.options = pypesto.HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
        )

        self.check_history()

    def test_trace_all(self):
        history = pypesto.MemoryHistory(options=pypesto.HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=True,
            trace_record_res=True,
            trace_record_sres=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        ))
        self.obj.history = history

        self.check_history()


class FunModeHistoryTest(HistoryTest):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = pypesto.ScipyOptimizer(
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

        self.obj.history.options = pypesto.objective.HistoryOptions(
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

        self.obj.history.options = pypesto.objective.HistoryOptions(
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

        self.obj.history.options = pypesto.objective.HistoryOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=True,
            trace_record_res=True,
            trace_record_sres=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )

        self.check_history()


@pytest.fixture(params=["", "memory", "csv"])
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

    if type(history) in \
            (pypesto.History, pypesto.Hdf5History):
        # TODO update as functionality is implemented
        with pytest.raises(NotImplementedError):
            history.get_grad_trace()
    else:
        grads = history.get_grad_trace()
        assert len(grads) == 10
        assert len(grads[0]) == 7

    if isinstance(history, (pypesto.MemoryHistory, pypesto.CsvHistory)):
        # TODO extend as functionality is implemented in other histories

        # assert x values are not all the same
        xs = np.array(history.get_x_trace())
        assert np.all(xs[:-1] != xs[-1])

        ress = history.get_res_trace()
        assert all(res is None for res in ress)
