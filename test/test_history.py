"""
This is for testing optimization of the pypesto.Objective.
"""


import numpy as np
import pypesto
import unittest
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
        self.obj.history.options = self.obj.options

        self.problem = pypesto.Problem(self.obj, self.lb, self.ub)

        optimize_options = pypesto.OptimizeOptions(
            allow_failed_starts=False
        )

        result = pypesto.minimize(
            problem=self.problem,
            optimizer=self.optimizer,
            n_starts=1,
            startpoint_method=pypesto.startpoint.uniform,
            options=optimize_options
        )
        # disable trace from here on
        self.obj.options.trace_record = False
        for start in result.optimize_result.list:
            it_final = int(start['trace'][('fval', np.NaN)].idxmin())
            it_start = int(np.where(np.logical_not(
                np.isnan(start['trace']['fval'].values)
            ))[0][0])
            self.assertTrue(np.isclose(
                start['trace']['x'].values[0, :], start['x0']
            ).all())
            self.assertTrue(np.isclose(
                start['trace']['x'].values[it_final, :], start['x']
            ).all())
            self.assertTrue(np.isclose(
                start['trace']['fval'].values[it_start, 0], start['fval0']
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
                        if not np.isnan(start['trace'].loc[it, (var, np.NaN)]):
                            self.assertTrue(np.isclose(
                                start['trace'][var].values[it, 0],
                                fun(start['trace']['x'].values[it, :])
                            ))
                    elif var in ['hess', 'sres', 'res']:
                        if start['trace'].loc[it, (var, np.NaN)] is not None:
                            self.assertTrue(np.isclose(
                                start['trace'][var].values[it, 0],
                                fun(start['trace']['x'].values[it, :])
                            ).all())
                    elif self.obj.options[f'trace_record_{var}'] and not \
                            np.isnan(start['trace'][var].values[it, :]).all():
                        self.assertTrue(np.isclose(
                            start['trace'][var].values[it, :],
                            fun(start['trace']['x'].values[it, :])
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
        self.obj.options = pypesto.objective.ObjectiveOptions(
            trace_record=True,
            trace_record_chi2=True,
            trace_record_schi2=False,
        )

        self.check_history()

    def test_trace_chi2_schi2(self):
        self.obj.options = pypesto.objective.ObjectiveOptions(
            trace_record=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )

        self.check_history()

    def test_trace_schi2(self):
        self.obj.options = pypesto.objective.ObjectiveOptions(
            trace_record=True,
            trace_record_chi2=True,
            trace_record_schi2=False,
        )

        self.check_history()

    def test_trace_grad(self):
        self.obj.options = pypesto.objective.ObjectiveOptions(
            trace_record=True,
            trace_record_grad=True,
        )

        self.check_history()

    def test_trace_all(self):
        self.obj.options = pypesto.objective.ObjectiveOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=True,
            trace_record_res=True,
            trace_record_sres=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )

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

        self.obj.options = pypesto.objective.ObjectiveOptions(
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

        self.obj.options = pypesto.objective.ObjectiveOptions(
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

        self.obj.options = pypesto.objective.ObjectiveOptions(
            trace_record=True,
            trace_record_grad=True,
            trace_record_hess=True,
            trace_record_res=True,
            trace_record_sres=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )

        self.check_history()
