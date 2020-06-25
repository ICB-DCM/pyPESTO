"""
This is for testing optimization of the pypesto.Objective.
"""


import numpy as np
import pytest
import test.test_objective as test_objective
import warnings
import re

import pypesto
import pypesto.optimize as optimize


@pytest.fixture(params=['separated', 'integrated'])
def mode(request):
    return request.param


optimizers = [
    *[('scipy', method) for method in [
        'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
        'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
        'trust-ncg', 'trust-exact', 'trust-krylov',
        'ls_trf', 'ls_dogbox']],
    # disabled: ,'trust-constr', 'ls_lm', 'dogleg'
    ('ipopt', ''),
    ('dlib', 'default'),
    ('pyswarm', ''),
]


@pytest.fixture(params=optimizers)
def optimizer(request):
    return request.param


def test_optimization(mode, optimizer):
    """Test optimization using various optimizers and objective modes."""
    if mode == 'separated':
        obj = test_objective.rosen_for_sensi(max_sensi_order=2,
                                             integrated=False)['obj']
    else:  # mode == 'integrated':
        obj = test_objective.rosen_for_sensi(max_sensi_order=2,
                                             integrated=True)['obj']

    library, method = optimizer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if re.match(r'^(?i)(ls_)', method):
            # obj has no residuals
            with pytest.raises(Exception):
                check_minimize(obj, library, method)
            # no error when allow failed starts
            check_minimize(obj, library, method, allow_failed_starts=True)
        else:
            check_minimize(obj, library, method)


def check_minimize(objective, library, solver, allow_failed_starts=False):

    options = {
        'maxiter': 100
    }

    optimizer = None

    if library == 'scipy':
        optimizer = optimize.ScipyOptimizer(method=solver, options=options)
    elif library == 'ipopt':
        optimizer = optimize.IpoptOptimizer()
    elif library == 'dlib':
        optimizer = optimize.DlibOptimizer(method=solver, options=options)
    elif library == 'pyswarm':
        optimizer = optimize.PyswarmOptimizer(options=options)

    lb = 0 * np.ones((1, 2))
    ub = 1 * np.ones((1, 2))
    problem = pypesto.Problem(objective, lb, ub)

    optimize_options = optimize.OptimizeOptions(
        allow_failed_starts=allow_failed_starts)

    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=1,
        startpoint_method=pypesto.startpoint.uniform,
        options=optimize_options
    )

    assert isinstance(result.optimize_result.list[0]['fval'], float)
