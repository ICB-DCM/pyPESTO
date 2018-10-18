"""
This is for testing the pypesto.Objective.
"""

import numpy as np
import scipy as sp
import pypesto
import unittest


class ObjectiveTest(unittest.TestCase):

    def test_evaluate(self):
        for obj in [get_objective_rosen_separated(),
                    get_objective_rosen_integrated()]:
            self._test_evaluate(obj, 
                                rosen_x, rosen_fval_true,
                                rosen_grad_true, rosen_hess_true)
        self._test_evaluate(get_objective_poly_integrated(),
                            poly_x, poly_fval_true,
                            poly_grad_true, poly_hess_true)

    def _test_evaluate(self, obj,
                       x, fval_true, grad_true, hess_true):

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

    def test_return_type(self):
        for obj in [get_objective_rosen_separated(),
                    get_objective_rosen_integrated()]:
            self._test_return_type(obj,
                                   rosen_x)
        self._test_return_type(get_objective_poly_integrated(), poly_x)

    def _test_return_type(self, obj, 
                          x):
        ret = obj(x, (0,))
        self.assertTrue(isinstance(ret, float))
        ret = obj(x, (1,))
        self.assertTrue(isinstance(ret, np.ndarray))
        ret = obj(x, (2,))
        self.assertTrue(isinstance(ret, np.ndarray))
        ret = obj(x, (0, 1))
        print(ret)
        self.assertTrue(isinstance(ret, tuple))
        self.assertTrue(len(ret) == 2)


rosen_x = np.array([0., 1.])
rosen_fval_true = 101.0
rosen_grad_true = [-2., 200.]
rosen_hess_true = [[-398., 0.],
                   [0., 200.]]


def get_objective_rosen_separated():
    return pypesto.Objective(fun=sp.optimize.rosen,
                             grad=sp.optimize.rosen_der,
                             hess=sp.optimize.rosen_hess)


def get_objective_rosen_integrated():
    def rosenbrock(x):
        return (sp.optimize.rosen(x),
                sp.optimize.rosen_der(x),
                sp.optimize.rosen_hess(x))
    return pypesto.Objective(fun=rosenbrock,
                             grad=True,
                             hess=True)


def get_objective_poly_integrated():
    def poly(x):
        return ((x-2)**2 + 1,
                2*(x-2),
                2)
    return pypesto.Objective(fun=poly, grad=True, hess=True)


# for testing in 1d
poly_x = 0.
poly_fval_true = 5.
poly_grad_true = -4
poly_hess_true = 2


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(ObjectiveTest())
    unittest.main()
