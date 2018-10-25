"""
This is for testing the pypesto.Objective.
"""

import numpy as np
import scipy as sp
import numbers
import pypesto
import unittest


class ObjectiveTest(unittest.TestCase):

    def test_evaluate(self):
        """
        Test if values are computed correctly.
        """
        for integrated in [True, False]:
            for struct in [_rosen_for_sensi(2, integrated, [0, 1]),
                           _poly_for_sensi(2, True, 0.5)]:
                self._test_evaluate(struct)

    def _test_evaluate(self, struct):
        obj = struct['obj']
        x = struct['x']
        fval_true = struct['fval']
        grad_true = struct['grad']
        hess_true = struct['hess']
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
            obj(x, (0, 1, 2))

        # check default argument
        self.assertTrue(np.isclose(obj(x), fval_true))

        # check convenience functions
        self.assertTrue(np.isclose(obj.get_fval(x), fval_true))
        if max_sensi_order >= 1:
            self.assertTrue(np.isclose(obj.get_grad(x), grad_true).all())
        if max_sensi_order >= 2:
            self.assertTrue(np.isclose(obj.get_hess(x), hess_true).all())

        # check different calling types
        if max_sensi_order >= 2:
            grad, hess = obj(x, (1, 2))
            self.assertTrue(np.isclose(grad, grad_true).all())
            self.assertTrue(np.isclose(hess, hess_true).all())

    def test_return_type(self):
        """
        Test if the output format is correct.
        """
        for integrated in [True, False]:
            for max_sensi_order in [2, 1, 0]:
                for struct in [_rosen_for_sensi(max_sensi_order,
                                                integrated, [0, 1]),
                               _poly_for_sensi(max_sensi_order,
                                               integrated, 0)]:
                    self._test_return_type(struct)

    def _test_return_type(self, struct):
        obj = struct['obj']
        x = struct['x']
        max_sensi_order = struct['max_sensi_order']

        ret = obj(x, (0,))
        self.assertTrue(isinstance(ret, numbers.Number))
        if max_sensi_order >= 1:
            ret = obj(x, (1,))
            self.assertTrue(isinstance(ret, np.ndarray))
        if max_sensi_order >= 2:
            ret = obj(x, (2,))
            self.assertTrue(isinstance(ret, np.ndarray))
        if max_sensi_order >= 1:
            ret = obj(x, (0, 1))
            self.assertTrue(isinstance(ret, tuple))
            self.assertTrue(len(ret) == 2)

    def test_sensis(self):
        """
        Test output when not all sensitivities can be computed.
        """
        for integrated in [True, False]:
            for max_sensi_order in [2, 1, 0]:
                for struct in [_rosen_for_sensi(max_sensi_order,
                                                integrated, [0, 1]),
                               _poly_for_sensi(max_sensi_order,
                                               integrated, 0)]:
                    self._test_sensis(struct)

    def _test_sensis(self, struct):
        obj = struct['obj']
        x = struct['x']
        max_sensi_order = struct['max_sensi_order']

        obj(x, (0,))
        if max_sensi_order >= 1:
            obj(x, (0, 1))
        else:
            with self.assertRaises(ValueError):
                obj(x, (0, 1))
        if max_sensi_order >= 2:
            obj(x, (0, 1, 2))
        else:
            with self.assertRaises(ValueError):
                obj(x, (0, 1, 2))


def _obj_for_sensi(fun, grad, hess, max_sensi_order, integrated, x):
    """
    Create a pypesto.Objective able to compute up to the speficied
    max_sensi_order. Returns a dict containing the objective obj as well
    as max_sensi_order and fval, grad, hess for the passed x.

    Parameters
    ----------

    fun, grad, hess: callable
        Functions computing the fval, grad, hess.
    max_sensi_order: int
        Maximum sensitivity order the pypesto.Objective should be capable of.
    integrated: bool
        True if fun, grad, hess should be integrated into one function, or
        passed to pypesto.Objective separately (both is possible)
    x: np.array
        Value at which to evaluate the function to obtain true values.

    Returns
    -------

    ret: dict
        With fields obj, max_sensi_order, x, fval, grad, hess.
    """
    if integrated:
        if max_sensi_order == 2:
            def arg_fun(x):
                return (fun(x), grad(x), hess(x))
            arg_grad = arg_hess = True
        elif max_sensi_order == 1:
            def arg_fun(x):
                return (fun(x), grad(x))
            arg_grad = True
            arg_hess = False
        else:
            def arg_fun(x):
                return fun(x)
            arg_grad = arg_hess = False
    else:  # integrated
        if max_sensi_order >= 2:
            arg_hess = hess
        else:
            arg_hess = None
        if max_sensi_order >= 1:
            arg_grad = grad
        else:
            arg_grad = None
        arg_fun = fun
    obj = pypesto.Objective(fun=arg_fun, grad=arg_grad, hess=arg_hess)
    return {'obj': obj,
            'max_sensi_order': max_sensi_order,
            'x': x,
            'fval': fun(x),
            'grad': grad(x),
            'hess': hess(x)}


def _rosen_for_sensi(max_sensi_order, integrated=False, x=[0, 1]):
    """
    Rosenbrock function from scipy.optimize.
    """
    return _obj_for_sensi(sp.optimize.rosen,
                          sp.optimize.rosen_der,
                          sp.optimize.rosen_hess,
                          max_sensi_order, integrated, x)


def _poly_for_sensi(max_sensi_order, integrated=False, x=0):
    """
    1-dim polynomial for testing in 1d.
    """

    def fun(x):
        return (x - 2)**2 + 1

    def grad(x):
        return 2 * (x - 2)

    def hess(x):
        return 2

    return _obj_for_sensi(fun, grad, hess,
                          max_sensi_order, integrated, x)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(ObjectiveTest())
    unittest.main()
