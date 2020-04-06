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


class HistoryTest(unittest.TestCase):
    problem: pypesto.Problem = None
    optimizer: pypesto.Optimizer = None
    obj: pypesto.Objective = None
    ub: np.ndarray = None
    lb: np.ndarray = None

    def check_history(self):
        self.problem = pypesto.Problem(self.obj, self.lb, self.ub)

        optimize_options = pypesto.OptimizeOptions(
            allow_failed_starts=False
        )

        history_options = pypesto.HistoryOptions(
            trace_record=True,
            trace_record_hess=False,
            trace_save_iter=1,
            storage_file='tmp/traces/conversion_example_{id}.csv',
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
        for start in result.optimize_result.list:
            trace = start.history._trace
            it_final = int(trace[('fval', np.NaN)].idxmin())
            it_start = int(np.where(np.logical_not(
                np.isnan(trace['fval'].values)
            ))[0][0])
            self.assertTrue(np.isclose(
                trace['x'].values[0, :], start.x0
            ).all())
            self.assertTrue(np.isclose(
                trace['x'].values[it_final, :], start.x
            ).all())
            self.assertTrue(np.isclose(
                trace['fval'].values[it_start, 0], start.fval0
            ))

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
                    if var in ['fval', 'chi2']:
                        if not np.isnan(trace[var].values[it, 0]):
                            self.assertTrue(np.isclose(
                                trace[var].values[it, 0],
                                fun(trace['x'].values[it, :])
                            ))
                    elif var in ['hess', 'sres', 'res']:
                        if trace[var].values[it, 0] is not None:
                            self.assertTrue(np.isclose(
                                trace[var].values[it, 0],
                                fun(trace['x'].values[it, :])
                            ).all())
                    elif self.obj.history.options[f'trace_record_{var}'] \
                            and not \
                            np.isnan(trace[var].values[it, :]).all():
                        self.assertTrue(np.isclose(
                            trace[var].values[it, :],
                            fun(trace['x'].values[it, :])
                        ).all())


class ResModeHistoryTest(HistoryTest):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = pypesto.ScipyOptimizer(
            method='ls_trf',
            options={'maxiter': 100}
        )
        cls.obj, _ = load_model_objective(
            'conversion_reaction'
        )

        cls.lb = -2 * np.ones((1, 2))
        cls.ub = 2 * np.ones((1, 2))

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
            (pypesto.History, pypesto.CsvHistory, pypesto.Hdf5History):
        # TODO update as functionality is implemented
        with pytest.raises(NotImplementedError):
            history.get_grad_trace()
    else:
        grads = history.get_grad_trace()
        assert len(grads) == 10
        assert len(grads[0]) == 7

    if type(history) == pypesto.MemoryHistory:
        # TODO extend as funcionality is implemented
        ress = history.get_res_trace()
        assert all(res is None for res in ress)
