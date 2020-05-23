"""
This is for testing the pypesto.Objective.
"""

import numpy as np
import pypesto
import pytest
from pypesto.objective.constants import MODE_RES

from .test_objective import poly_for_sensi, rosen_for_sensi
from .test_sbml_conversion import load_model_objective

ATOL = 1e-6
RTOL = 1e-4


def convreact_for_funmode(max_sensi_order, x=None):
    obj = load_model_objective('conversion_reaction')[0]
    return {'obj': obj,
            'max_sensi_order': max_sensi_order,
            'x': x,
            'fval': obj.get_fval(x),
            'grad': obj.get_grad(x),
            'hess': obj.get_hess(x)}


def convreact_for_resmode(max_sensi_order, x=None):
    obj = load_model_objective('conversion_reaction')[0]
    return {'obj': obj,
            'max_sensi_order': max_sensi_order,
            'x': x,
            'res': obj.get_res(x),
            'sres': obj.get_sres(x)}


def test_evaluate():
    """
    Test if values are computed correctly.
    """
    for struct in [rosen_for_sensi(2, False, [0, 1]),
                   poly_for_sensi(2, True, 0.5),
                   convreact_for_funmode(2, [-0.3, -0.7])]:
        _test_evaluate_funmode(struct)

    _test_evaluate_resmode(convreact_for_resmode(1, [-0.3, -0.7]))


def _test_evaluate_funmode(struct):
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
        assert np.isclose(fval, fval_true, atol=ATOL, rtol=RTOL)
        assert np.isclose(grad, grad_true, atol=ATOL, rtol=RTOL).all()
        assert np.isclose(hess, hess_true, atol=ATOL, rtol=RTOL).all()
    elif max_sensi_order >= 1:
        fval, grad = obj(x, (0, 1))
        assert np.isclose(fval, fval_true, atol=ATOL, rtol=RTOL)
        assert np.isclose(grad, grad_true, atol=ATOL, rtol=RTOL).all()

    # check default argument
    assert np.isclose(obj(x), fval_true, atol=ATOL, rtol=RTOL)

    # check convenience functions
    assert np.isclose(obj.get_fval(x), fval_true, atol=ATOL, rtol=RTOL)
    if max_sensi_order >= 1:
        assert np.isclose(obj.get_grad(x), grad_true,
                          atol=ATOL, rtol=RTOL).all()
    if max_sensi_order >= 2:
        assert np.isclose(obj.get_hess(x), hess_true,
                          atol=ATOL, rtol=RTOL).all()

    # check different calling types
    if max_sensi_order >= 1:
        grad = obj(x, (1,))
        assert np.isclose(grad, grad_true).all()

    if max_sensi_order >= 2:
        grad, hess = obj(x, (1, 2))
        assert np.isclose(grad, grad_true, atol=ATOL, rtol=RTOL).all()
        assert np.isclose(hess, hess_true, atol=ATOL, rtol=RTOL).all()

        hess = obj(x, (2,))
        assert np.isclose(hess, hess_true, atol=ATOL, rtol=RTOL).all()


def _test_evaluate_resmode(struct):
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
        assert np.isclose(res, res_true, atol=ATOL, rtol=RTOL).all()
        assert np.isclose(sres, sres_true, atol=ATOL, rtol=RTOL).all()

    res = obj(x, (0,), MODE_RES)
    assert np.isclose(res, res_true, atol=ATOL, rtol=RTOL).all()

    # check convenience functions)
    assert np.isclose(obj.get_res(x), res_true, atol=ATOL, rtol=RTOL).all()
    if max_sensi_order >= 1:
        assert np.isclose(obj.get_sres(x), sres_true,
                          atol=ATOL, rtol=RTOL).all()


def test_exceptions():
    with pytest.raises(TypeError):
        pypesto.objective.AggregatedObjective(
            rosen_for_sensi(2, False, [0, 1])['obj'])
    with pytest.raises(TypeError):
        pypesto.objective.AggregatedObjective([0.5])

    with pytest.raises(ValueError):
        pypesto.objective.AggregatedObjective([])
