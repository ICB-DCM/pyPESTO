"""
This is for testing the pypesto.Objective.
"""

import numpy as np
import pypesto
import unittest
from pypesto.objective.constants import MODE_RES

from .test_objective import poly_for_sensi, rosen_for_sensi
from .test_sbml_conversion import _load_model_objective


def convreact_for_funmode(max_sensi_order, x=None):
    obj = _load_model_objective('conversion_reaction')[0]
    return {'obj': obj,
            'max_sensi_order': max_sensi_order,
            'x': x,
            'fval': obj.get_fval(x),
            'grad': obj.get_grad(x),
            'hess': obj.get_hess(x)}


def convreact_for_resmode(max_sensi_order, x=None):
    obj = _load_model_objective('conversion_reaction')[0]
    return {'obj': obj,
            'max_sensi_order': max_sensi_order,
            'x': x,
            'res': obj.get_res(x),
            'sres': obj.get_sres(x)}


class AggregateObjectiveTest(unittest.TestCase):

    def test_evaluate(self):
        """
        Test if values are computed correctly.
        """
        for struct in [rosen_for_sensi(2, False, [0, 1]),
                       poly_for_sensi(2, True, 0.5),
                       convreact_for_funmode(2, [-0.3, -0.7])]:
            self._test_evaluate_funmode(struct)

        self._test_evaluate_resmode(convreact_for_resmode(1, [-0.3, -0.7]))

    def _test_evaluate_funmode(self, struct):
        obj = pypesto.objective.AggregatedObjective(
            [struct['obj'], struct['obj']]
        )
        x = struct['x']
        fval_true = 2*struct['fval']
        grad_true = 2*struct['grad']
        hess_true = 2*struct['hess']
        max_sensi_order = struct['max_sensi_order']

        # check function values
        if max_sensi_order >= 2:
            fval, grad, hess = obj(x, (0, 1, 2))
            self.assertTrue(np.isclose(fval, fval_true))
            self.assertTrue(np.isclose(grad, grad_true).all())
            self.assertTrue(np.isclose(hess, hess_true).all())
        elif max_sensi_order >= 1:
            fval, grad = obj(x, (0, 1))
            self.assertTrue(np.isclose(fval, fval_true))
            self.assertTrue(np.isclose(grad, grad_true).all())

        # check default argument
        self.assertTrue(np.isclose(obj(x), fval_true))

        # check convenience functions
        self.assertTrue(np.isclose(obj.get_fval(x), fval_true))
        if max_sensi_order >= 1:
            self.assertTrue(np.isclose(obj.get_grad(x), grad_true).all())
        if max_sensi_order >= 2:
            self.assertTrue(np.isclose(obj.get_hess(x), hess_true).all())

        # check different calling types
        if max_sensi_order >= 1:
            grad = obj(x, (1,))
            self.assertTrue(np.isclose(grad, grad_true).all())

        if max_sensi_order >= 2:
            grad, hess = obj(x, (1, 2))
            self.assertTrue(np.isclose(grad, grad_true).all())
            self.assertTrue(np.isclose(hess, hess_true).all())

            hess = obj(x, (2,))
            self.assertTrue(np.isclose(hess, hess_true).all())

    def _test_evaluate_resmode(self, struct):
        obj = pypesto.objective.AggregatedObjective(
            [struct['obj'], struct['obj']]
        )
        x = struct['x']
        res_true = np.hstack([struct['res'], struct['res']])
        sres_true = np.vstack([struct['sres'], struct['sres']])
        max_sensi_order = struct['max_sensi_order']

        # check function values
        if max_sensi_order >= 1:
            res, sres = obj(x, (0, 1), MODE_RES)
            self.assertTrue(np.isclose(res, res_true).all())
            self.assertTrue(np.isclose(sres, sres_true).all())

        res = obj(x, (0,), MODE_RES)
        self.assertTrue(np.isclose(res, res_true).all())

        # check convenience functions
        self.assertTrue(np.isclose(obj.get_res(x), res_true).all())
        if max_sensi_order >= 1:
            self.assertTrue(np.isclose(obj.get_sres(x), sres_true).all())

    def test_exceptions(self):
        self.assertRaises(
            TypeError,
            pypesto.objective.AggregatedObjective,
            rosen_for_sensi(2, False, [0, 1])['obj']
        )
        self.assertRaises(
            TypeError,
            pypesto.objective.AggregatedObjective,
            [0.5]
        )
        self.assertRaises(
            ValueError,
            pypesto.objective.AggregatedObjective,
            [
                rosen_for_sensi(2, False, [0, 1])['obj'],
                rosen_for_sensi(2, True, [0, 1])['obj']
            ]
        )
        self.assertRaises(
            ValueError,
            pypesto.objective.AggregatedObjective,
            []
        )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(AggregateObjectiveTest())
    unittest.main()
