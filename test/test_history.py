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
            # the max here is arbitrary, one of the two will be NaN and we
            # want to select the other one
            it_final = start['trace'][[('fval', np.NaN),
                                       ('chi2', np.NaN)]].idxmin().max()
            self.assertTrue(np.array_equal(
                start['trace'].loc[0, 'x'].values, start['x0']
            ))
            self.assertTrue(np.array_equal(
                start['trace'].loc[it_final, 'x'].values, start['x']
            ))

            funs = {
                'fval': self.obj.get_fval,
                'grad': self.obj.get_grad,
                'hess': self.obj.get_hess,
                'chi2': lambda x: res_to_chi2(self.obj.get_res(x)),
                'schi2': lambda x: sres_to_schi2(*self.obj(
                    start['trace'].loc[it, 'x'].values,
                    (0, 1,),
                    pypesto.objective.constants.MODE_RES
                ))
            }
            for var, fun in funs.items():
                for it in range(5):
                    if var in ['fval', 'chi2']:
                        if not np.isnan(start['trace'].loc[it, (var, np.NaN)]):
                            self.assertEqual(
                                start['trace'].loc[it, (var, np.NaN)],
                                fun(start['trace'].loc[it, 'x'].values)
                            )
                    elif self.obj.options[f'trace_record_{var}'] and not \
                            np.isnan(start['trace'].loc[it, var].values).all():
                        self.assertTrue(np.isclose(
                            start['trace'].loc[it, var].values,
                            fun(start['trace'].loc[it, 'x'].values)
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
        self.obj.history.options = self.obj.options

        self.check_history()

    def test_trace_chi2_schi2(self):
        self.obj.options = pypesto.objective.ObjectiveOptions(
            trace_record=True,
            trace_record_chi2=True,
            trace_record_schi2=True,
        )
        self.obj.history.options = self.obj.options

        self.check_history()

    def test_trace_schi2(self):
        self.obj.options = pypesto.objective.ObjectiveOptions(
            trace_record=True,
            trace_record_chi2=True,
            trace_record_schi2=False,
        )
        self.obj.history.options = self.obj.options

        self.check_history()

    def test_trace_grad(self):
        self.obj.options = pypesto.objective.ObjectiveOptions(
            trace_record=True,
            trace_record_grad=True,
        )
        self.obj.history.options = self.obj.options

        self.check_history()


class FunModeHistoryTest(HistoryTest):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = pypesto.ScipyOptimizer(
            method='L-BFGS-B',
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
        self.obj.history.options = self.obj.options

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
        self.obj.history.options = self.obj.options

        self.check_history()

    def test_trace_chi2(self):
        self.obj = rosen_for_sensi(
            max_sensi_order=2,
            integrated=True
        )['obj']

        self.obj.options = pypesto.objective.ObjectiveOptions(
            trace_record=True,
            trace_record_chi2=True,
        )
        self.obj.history.options = self.obj.options

        self.check_history()

