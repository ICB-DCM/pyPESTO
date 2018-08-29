"""
This is for testing the pypesto.Objective.
"""

import numpy as np
import scipy as sp
import pypesto
import unittest


class ObjectiveTest(unittest.TestCase):

    def test_objective_separated(self):
        obj = get_objective_rosen_separated()
        self.check_evaluation(obj)

    def test_objective_integrated(self):
        obj = get_objective_rosen_integrated()
        self.check_evaluation(obj)

    def check_evaluation(self, obj):
        x = np.array([0, 1])

        fval_true = 101.0
        grad_true = [-2, 200]
        hess_true = [[-398, 0], [0, 200]]

        # check function values
        fval, grad, hess = obj(x, (0, 1, 2))
        self.assertTrue(np.isclose(fval, fval_true))
        self.assertTrue(np.isclose(grad, grad_true).all())
        self.assertTrue(np.isclose(hess, hess_true).all())

        # check default argument
        self.assertTrue(np.isclose(obj(x), fval_true))

        # check convenience functions
        self.assertTrue(np.isclose(obj.get_fval(x), fval_true))
        self.assertTrue(np.isclose(obj.get_grad(x), grad_true).all())
        self.assertTrue(np.isclose(obj.get_hess(x), hess_true).all())

        # check different calling types
        grad, hess = obj(x, (1, 2))
        self.assertTrue(np.isclose(grad, grad_true).all())
        self.assertTrue(np.isclose(hess, hess_true).all())


def get_objective_rosen_separated():
    return pypesto.Objective(fun=sp.optimize.rosen,
                             grad=sp.optimize.rosen_der,
                             hess=sp.optimize.rosen_hess,
                             dim=2)


def get_objective_rosen_integrated():
    def rosenbrock(x):
        return (sp.optimize.rosen(x),
                sp.optimize.rosen_der(x),
                sp.optimize.rosen_hess(x))
    return pypesto.Objective(fun=rosenbrock,
                             grad=True,
                             hess=True,
                             dim=2)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(ObjectiveTest())
    unittest.main()
