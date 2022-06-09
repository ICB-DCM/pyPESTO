"""Test the :class:`pypesto.Objective`."""

import copy
import numbers

import aesara.tensor as aet
import numpy as np
import pytest
import sympy as sp

import pypesto
from pypesto.objective.aesara import AesaraObjective

from ..util import CRProblem, poly_for_sensi, rosen_for_sensi


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
    for struct in [
        rosen_for_sensi(2, integrated, [0, 1]),
        poly_for_sensi(2, True, 0.5),
    ]:
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
    for struct in [
        rosen_for_sensi(max_sensi_order, integrated, [0, 1]),
        poly_for_sensi(max_sensi_order, integrated, 0),
    ]:
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
    for struct in [
        rosen_for_sensi(max_sensi_order, integrated, [0, 1]),
        poly_for_sensi(max_sensi_order, integrated, 0),
    ]:
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


def test_finite_difference_checks():
    """
    Test the finite difference gradient check methods by expected relative
    error.
    """
    x = sp.Symbol('x')

    # Setup single-parameter objective function
    fun_expr = x**10
    grad_expr = fun_expr.diff()
    theta = 0.1

    fun = sp.lambdify(x, fun_expr)
    grad = sp.lambdify(x, grad_expr)

    objective = pypesto.Objective(fun=fun, grad=grad)

    def rel_err(eps_):
        """Expected relative error."""
        central_difference = (fun(theta + eps_) - fun(theta - eps_)) / (
            2 * eps_
        )
        return abs(
            (grad(theta) - central_difference) / (central_difference + eps_)
        )

    # Test the single step size `check_grad` method.
    eps = 1e-5
    result_single_eps = objective.check_grad(np.array([theta]), eps=eps)
    assert result_single_eps['rel_err'].squeeze() == rel_err(eps)

    # Test the multiple step size `check_grad_multi_eps` method.
    multi_eps = {1e-1, 1e-3, 1e-5, 1e-7, 1e-9}
    result_multi_eps = objective.check_grad_multi_eps(
        [theta], multi_eps=multi_eps
    )
    assert result_multi_eps['rel_err'].squeeze() == min(
        rel_err(_eps) for _eps in multi_eps
    )


def test_aesara(max_sensi_order, integrated):
    """Test function composition and gradient computation via aesara"""
    prob = rosen_for_sensi(max_sensi_order, integrated, [0, 1])

    # create aesara specific symbolic tensor variables
    x = aet.specify_shape(aet.vector('x'), (2,))

    # apply inverse transform such that we evaluate at prob['x']
    x_ref = np.arcsinh(prob['x'])

    # compose rosenbrock function with with sinh transformation
    obj = AesaraObjective(prob['obj'], x, aet.sinh(x))

    # check value against
    assert obj(x_ref) == prob['fval']

    if max_sensi_order > 0:
        assert np.allclose(
            obj(x_ref, sensi_orders=(1,)), prob['grad'] * np.cosh(x_ref)
        )

    if max_sensi_order > 1:
        assert np.allclose(
            prob['hess'] * (np.diag(np.power(np.cosh(x_ref), 2)))
            + np.diag(prob['grad'] * np.sinh(x_ref)),
            obj(x_ref, sensi_orders=(2,)),
        )

    # test everything still works after deepcopy
    cobj = copy.deepcopy(obj)
    assert cobj(x_ref) == prob['fval']


@pytest.fixture(
    params=[pypesto.FD.CENTRAL, pypesto.FD.FORWARD, pypesto.FD.BACKWARD]
)
def fd_method(request) -> str:
    """Finite difference method."""
    return request.param


@pytest.fixture(
    params=[
        1e-6,
        pypesto.FDDelta.CONSTANT,
        pypesto.FDDelta.DISTANCE,
        pypesto.FDDelta.STEPS,
        pypesto.FDDelta.ALWAYS,
    ]
)
def fd_delta(request):
    """Finite difference step size method."""
    return request.param


def test_fds(fd_method, fd_delta):
    """Test finite differences."""
    problem = CRProblem()

    # reference objective
    obj = problem.get_objective()

    # FDs for everything
    obj_fd = pypesto.FD(
        obj,
        grad=True,
        hess=True,
        sres=True,
        method=fd_method,
        delta_fun=fd_delta,
        delta_grad=fd_delta,
        delta_res=fd_delta,
    )
    # bases Hessian on gradients
    obj_fd_grad = pypesto.FD(
        obj,
        grad=True,
        hess=True,
        sres=True,
        hess_via_fval=False,
        method=fd_method,
        delta_fun=fd_delta,
        delta_grad=fd_delta,
        delta_res=fd_delta,
    )
    # does not actually use FDs
    obj_fd_fake = pypesto.FD(
        obj,
        grad=None,
        hess=None,
        sres=None,
        method=fd_method,
        delta_fun=fd_delta,
        delta_grad=fd_delta,
        delta_res=fd_delta,
    )
    # limited outputs, no derivatives
    obj_fd_limited = pypesto.FD(
        obj,
        grad=False,
        hess=False,
        sres=False,
        method=fd_method,
        delta_fun=fd_delta,
        delta_grad=fd_delta,
        delta_res=fd_delta,
    )
    p = problem.p_true

    # check that function values coincide (call delegated)
    for attr in ['fval', 'res']:
        val = getattr(obj, f"get_{attr}")(p)
        val_fd = getattr(obj_fd, f"get_{attr}")(p)
        val_fd_grad = getattr(obj_fd_grad, f"get_{attr}")(p)
        val_fd_fake = getattr(obj_fd_fake, f"get_{attr}")(p)
        val_fd_limited = getattr(obj_fd_limited, f"get_{attr}")(p)
        assert (
            (val == val_fd).all()
            and (val == val_fd_grad).all()
            and (val == val_fd_fake).all()
            and (val == val_fd_limited).all()
        ), attr

    # check that derivatives are close
    if fd_method == pypesto.FD.CENTRAL:
        atol = rtol = 1e-4
    else:
        atol = rtol = 1e-2
    for attr in ['grad', 'hess', 'sres']:
        val = getattr(obj, f"get_{attr}")(p)
        val_fd = getattr(obj_fd, f"get_{attr}")(p)
        val_fd_grad = getattr(obj_fd_grad, f"get_{attr}")(p)
        val_fd_fake = getattr(obj_fd_fake, f"get_{attr}")(p)

        assert np.allclose(val, val_fd, atol=atol, rtol=rtol), attr
        # cannot completely coincide
        assert (val != val_fd).any(), attr

        assert np.allclose(val, val_fd_grad, atol=atol, rtol=rtol), attr
        # cannot completely coincide
        assert (val != val_fd_grad).any(), attr

        if attr == 'hess':
            assert (val_fd != val_fd_grad).any(), attr
        # should use available actual functionality
        assert (val == val_fd_fake).all(), attr

        # cannot be called
        with pytest.raises(ValueError):
            getattr(obj_fd_limited, f"get_{attr}")(p)

    # evaluate a couple times and assert number of update steps is as expected
    for i in range(31):
        obj_fd(10 * i * p, sensi_orders=(0, 1))
    if fd_delta == pypesto.FDDelta.CONSTANT:
        assert obj_fd.delta_fun.updates == 1
    elif isinstance(fd_delta, (float, np.ndarray)):
        assert obj_fd.delta_fun.updates == 0
    else:
        assert obj_fd.delta_fun.updates > 1
