"""
This is for testing optimization of the pypesto.Objective.
"""

import numpy as np
import pytest
import warnings
import re
import nlopt
import fides
import itertools as itt
import os
# import subprocess

import pypesto
import pypesto.optimize as optimize

from ..util import rosen_for_sensi


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
    ('dlib', ''),
    ('pyswarm', ''),
    ('cmaes', ''),
    ('scipydiffevolopt', ''),
    *[('nlopt', method) for method in [
        nlopt.LD_VAR1, nlopt.LD_VAR2, nlopt.LD_TNEWTON_PRECOND_RESTART,
        nlopt.LD_TNEWTON_PRECOND, nlopt.LD_TNEWTON_RESTART,
        nlopt.LD_TNEWTON, nlopt.LD_LBFGS,
        nlopt.LD_SLSQP, nlopt.LD_CCSAQ, nlopt.LD_MMA, nlopt.LN_SBPLX,
        nlopt.LN_NELDERMEAD, nlopt.LN_PRAXIS, nlopt.LN_NEWUOA,
        nlopt.LN_NEWUOA_BOUND, nlopt.LN_BOBYQA, nlopt.LN_COBYLA,
        nlopt.GN_ESCH, nlopt.GN_ISRES, nlopt.GN_AGS, nlopt.GD_STOGO,
        nlopt.GD_STOGO_RAND, nlopt.G_MLSL, nlopt.G_MLSL_LDS, nlopt.GD_MLSL,
        nlopt.GD_MLSL_LDS, nlopt.GN_CRS2_LM, nlopt.GN_ORIG_DIRECT,
        nlopt.GN_ORIG_DIRECT_L, nlopt.GN_DIRECT, nlopt.GN_DIRECT_L,
        nlopt.GN_DIRECT_L_NOSCAL, nlopt.GN_DIRECT_L_RAND,
        nlopt.GN_DIRECT_L_RAND_NOSCAL, nlopt.AUGLAG, nlopt.AUGLAG_EQ
    ]],
    *[('fides', solver) for solver in itt.product(
        [None, fides.SR1(), fides.BFGS(), fides.DFP()],
        [fides.SubSpaceDim.FULL, fides.SubSpaceDim.TWO]
    )]
]


@pytest.fixture(params=optimizers)
def optimizer(request):
    return request.param


def test_optimization(mode, optimizer):
    """Test optimization using various optimizers and objective modes."""
    if mode == 'separated':
        obj = rosen_for_sensi(max_sensi_order=2, integrated=False)['obj']
    else:  # mode == 'integrated':
        obj = rosen_for_sensi(max_sensi_order=2, integrated=True)['obj']

    library, method = optimizer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(method, str) and re.match(r'^(?i)(ls_)', method):
            # obj has no residuals
            with pytest.raises(Exception):
                check_minimize(obj, library, method)
            # no error when allow failed starts
            check_minimize(obj, library, method, allow_failed_starts=True)
        else:
            check_minimize(obj, library, method)


def test_unbounded_minimize(optimizer):
    """
    Test unbounded optimization using various optimizers and objective modes.
    """
    lb_init = 1.1 * np.ones((1, 2))
    lb = -np.inf * np.ones(lb_init.shape)
    ub_init = 1.11 * np.ones((1, 2))
    ub = np.inf * np.ones(ub_init.shape)
    problem = pypesto.Problem(
        rosen_for_sensi(max_sensi_order=2)['obj'],
        lb, ub, lb_init=lb_init, ub_init=ub_init
    )
    opt = get_optimizer(*optimizer)

    options = optimize.OptimizeOptions(allow_failed_starts=False)

    if isinstance(optimizer[1], str) and re.match(r'^(?i)(ls_)', optimizer[1]):
        return

    if optimizer in [('dlib', ''), ('pyswarm', ''), ('cmaes', ''),
                     ('scipydiffevolopt', ''),
                     *[('nlopt', method) for method in [
                         nlopt.GN_ESCH, nlopt.GN_ISRES, nlopt.GN_AGS,
                         nlopt.GD_STOGO, nlopt.GD_STOGO_RAND, nlopt.G_MLSL,
                         nlopt.G_MLSL_LDS, nlopt.GD_MLSL, nlopt.GD_MLSL_LDS,
                         nlopt.GN_CRS2_LM, nlopt.GN_ORIG_DIRECT,
                         nlopt.GN_ORIG_DIRECT_L, nlopt.GN_DIRECT,
                         nlopt.GN_DIRECT_L, nlopt.GN_DIRECT_L_NOSCAL,
                         nlopt.GN_DIRECT_L_RAND,
                         nlopt.GN_DIRECT_L_RAND_NOSCAL]]]:
        with pytest.raises(ValueError):
            optimize.minimize(
                problem=problem,
                optimizer=opt,
                n_starts=1,
                startpoint_method=pypesto.startpoint.uniform,
                options=options
            )
        return
    else:
        result = optimize.minimize(
            problem=problem,
            optimizer=opt,
            n_starts=1,
            startpoint_method=pypesto.startpoint.uniform,
            options=options
        )

    # check that ub/lb were reverted
    assert isinstance(result.optimize_result.list[0]['fval'], float)
    if optimizer not in [('scipy', 'ls_trf'), ('scipy', 'ls_dogbox')]:
        assert np.isfinite(result.optimize_result.list[0]['fval'])
        assert result.optimize_result.list[0]['x'] is not None
    # check that result is not in bounds, optimum is at (1,1), so you would
    # hope that any reasonable optimizer manage to finish with x < ub,
    # but I guess some are pretty terrible
    assert np.any(result.optimize_result.list[0]['x'] < lb_init) or \
        np.any(result.optimize_result.list[0]['x'] > ub_init)


def get_optimizer(library, solver):
    """Constructs Optimizer given and optimization library and optimization
    solver specification"""
    options = {
        'maxiter': 100
    }
    optimizer = None

    if library == 'scipy':
        optimizer = optimize.ScipyOptimizer(method=solver, options=options)
    elif library == 'ipopt':
        optimizer = optimize.IpoptOptimizer()
    elif library == 'dlib':
        optimizer = optimize.DlibOptimizer(options=options)
    elif library == 'pyswarm':
        optimizer = optimize.PyswarmOptimizer(options=options)
    elif library == 'cmaes':
        optimizer = optimize.CmaesOptimizer(options=options)
    elif library == 'scipydiffevolopt':
        optimizer = optimize.ScipyDifferentialEvolutionOptimizer(
            options=options)
    elif library == 'nlopt':
        optimizer = optimize.NLoptOptimizer(method=solver, options=options)
    elif library == 'fides':
        options[fides.Options.SUBSPACE_DIM] = solver[1]
        optimizer = optimize.FidesOptimizer(options=options,
                                            hessian_update=solver[0])

    return optimizer


def check_minimize(objective, library, solver, allow_failed_starts=False):
    """Runs a single run of optimization according to the provided inputs
    and checks whether optimization yielded a solution."""
    optimizer = get_optimizer(library, solver)
    lb = 0 * np.ones((1, 2))
    ub = 1 * np.ones((1, 2))
    problem = pypesto.Problem(objective, lb, ub)

    optimize_options = optimize.OptimizeOptions(
        allow_failed_starts=allow_failed_starts
    )

    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=1,
        startpoint_method=pypesto.startpoint.uniform,
        options=optimize_options
    )

    assert isinstance(result.optimize_result.list[0]['fval'], float)
    if (library, solver) not in [
            ('scipy', 'ls_trf'),
            ('scipy', 'ls_dogbox'),
            ('nlopt', nlopt.GD_STOGO_RAND)  # id 9, fails in 40% of cases
    ]:
        assert np.isfinite(result.optimize_result.list[0]['fval'])
        assert result.optimize_result.list[0]['x'] is not None


def test_MPIPool_Engine():
    """
    Test the MPIPoolEngine by calling an example script with mpiexec.
    """
    # get the path to this file:
    path = os.path.dirname(__file__)
    # run the example file.
    ret = os.system(f"mpiexec -np 2 python -m mpi4py.futures "
                    f"{path}../../doc/example/example_MPIPool.py")
    if ret != 0:
        raise Exception("The MPIPoolEngine test failed.")
