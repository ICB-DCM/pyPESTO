"""
This is for testing the pypesto.Objective.
"""

import numpy as np
import scipy.optimize as so
import numbers
import pytest
import pypesto


@pytest.fixture(params=[True, False])
def integrated(request):
    return request.param


@pytest.fixture(params=[2, 1, 0])
def max_sensi_order(request):
    return request.param


def test_evaluate(integrated):
    """
    Test if values are computed correctly.
    """
    for struct in [rosen_for_sensi(2, integrated, [0, 1]),
                   poly_for_sensi(2, True, 0.5)]:
        _test_evaluate(struct)


def _test_evaluate(struct):
    obj = struct['obj']
    x = struct['x']
    fval_true = struct['fval']
    grad_true = struct['grad']
    hess_true = struct['hess']
    max_sensi_order = struct['max_sensi_order']

    # check function values
    if max_sensi_order >= 2:
        fval, grad, hess = obj(x, (0, 1, 2))
        assert np.isclose(fval, fval_true)
        assert np.isclose(grad, grad_true).all()
        assert np.isclose(hess, hess_true).all()
    elif max_sensi_order >= 1:
        fval, grad = obj(x, (0, 1))
        assert np.isclose(fval, fval_true)
        assert np.isclose(grad, grad_true).all()
        obj(x, (0, 1, 2))

    # check default argument
    assert np.isclose(obj(x), fval_true)

    # check convenience functions
    assert np.isclose(obj.get_fval(x), fval_true)
    if max_sensi_order >= 1:
        assert np.isclose(obj.get_grad(x), grad_true).all()
    if max_sensi_order >= 2:
        assert np.isclose(obj.get_hess(x), hess_true).all()

    # check different calling types
    if max_sensi_order >= 2:
        grad, hess = obj(x, (1, 2))
        assert np.isclose(grad, grad_true).all()
        assert np.isclose(hess, hess_true).all()


def test_return_type(integrated, max_sensi_order):
    """
    Test if the output format is correct.
    """
    for struct in [rosen_for_sensi(max_sensi_order,
                                   integrated, [0, 1]),
                   poly_for_sensi(max_sensi_order,
                                  integrated, 0)]:
        _test_return_type(struct)


def _test_return_type(struct):
    obj = struct['obj']
    x = struct['x']
    max_sensi_order = struct['max_sensi_order']

    ret = obj(x, (0,))
    assert isinstance(ret, numbers.Number)
    if max_sensi_order >= 1:
        ret = obj(x, (1,))
        assert isinstance(ret, np.ndarray)
    if max_sensi_order >= 2:
        ret = obj(x, (2,))
        assert isinstance(ret, np.ndarray)
    if max_sensi_order >= 1:
        ret = obj(x, (0, 1))
        assert isinstance(ret, tuple)
        assert len(ret) == 2


def test_sensis(integrated, max_sensi_order):
    """
    Test output when not all sensitivities can be computed.
    """
    for struct in [rosen_for_sensi(max_sensi_order,
                                   integrated, [0, 1]),
                   poly_for_sensi(max_sensi_order,
                                  integrated, 0)]:
        _test_sensis(struct)


def _test_sensis(struct):
    obj = struct['obj']
    x = struct['x']
    max_sensi_order = struct['max_sensi_order']

    obj(x, (0,))
    if max_sensi_order >= 1:
        obj(x, (0, 1))
    else:
        with pytest.raises(ValueError):
            obj(x, (0, 1))
    if max_sensi_order >= 2:
        obj(x, (0, 1, 2))
    else:
        with pytest.raises(ValueError):
            obj(x, (0, 1, 2))


def obj_for_sensi(fun, grad, hess, max_sensi_order, integrated, x):
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
                return fun(x), grad(x), hess(x)
            arg_grad = arg_hess = True
        elif max_sensi_order == 1:
            def arg_fun(x):
                return fun(x), grad(x)
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


def rosen_for_sensi(max_sensi_order, integrated=False, x=None):
    """
    Rosenbrock function from scipy.optimize.
    """
    if x is None:
        x = [0, 1]

    return obj_for_sensi(so.rosen,
                         so.rosen_der,
                         so.rosen_hess,
                         max_sensi_order, integrated, x)


def poly_for_sensi(max_sensi_order, integrated=False, x=0.):
    """
    1-dim polynomial for testing in 1d.
    """

    def fun(x):
        return (x - 2)**2 + 1

    def grad(x):
        return 2 * (x - 2)

    def hess(_):
        return 2

    return obj_for_sensi(fun, grad, hess,
                         max_sensi_order, integrated, x)
