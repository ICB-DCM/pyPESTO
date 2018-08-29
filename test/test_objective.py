"""
This is for testing the pypesto.Objective.
"""

import numpy as np
import scipy as sp
import pypesto
import unittest


class ObjectiveTest(unittest.TestCase):
    def runTest(self):
        for mode in ['seperated', 'integrated']:
            with self.subTest(mode=mode):
                if mode == 'seperated':
                    test_objective_separated()
                elif mode == 'integrated':
                    test_objective_integrated()


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


def test_objective_separated():
    obj = get_objective_rosen_separated()
    _test_rosenbrock_objective(obj)


def test_objective_integrated():
    obj = get_objective_rosen_integrated()
    _test_rosenbrock_objective(obj)


def _test_rosenbrock_objective(obj):
    x = np.array([0, 1])

    fval_true = 101.0
    grad_true = [-2, 200]
    hess_true = [[-398, 0], [0, 200]]

    # check function values
    fval, grad, hess = obj(x, (0, 1, 2))
    assert np.isclose(fval, fval_true)
    assert np.isclose(grad, grad_true).all()
    assert np.isclose(hess, hess_true).all()

    # check default argument
    assert np.isclose(obj(x), fval_true)

    # check convenience functions
    assert np.isclose(obj.get_fval(x), fval_true)
    assert np.isclose(obj.get_grad(x), grad_true).all()
    assert np.isclose(obj.get_hess(x), hess_true).all()

    # check different calling types
    grad, hess = obj(x, (1, 2))
    assert np.isclose(grad, grad_true).all()
    assert np.isclose(hess, hess_true).all()


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(ObjectiveTest())
    unittest.main()
