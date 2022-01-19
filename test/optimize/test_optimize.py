"""
This is for testing optimization of the pypesto.Objective.
"""

import itertools as itt
import os
import re
import subprocess  # noqa: S404
import warnings

import fides
import nlopt
import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_almost_equal

import pypesto
import pypesto.optimize as optimize
from pypesto.store import read_result

from ..util import CRProblem, rosen_for_sensi


@pytest.fixture(params=['cr', 'rosen-integrated', 'rosen-separated'])
def problem(request) -> pypesto.Problem:
    if request.param == 'cr':
        return CRProblem().get_problem()
    elif 'rosen' in request.param:
        integrated = 'integrated' in request.param
        obj = rosen_for_sensi(max_sensi_order=2, integrated=integrated)['obj']
        lb = 0 * np.ones((1, 2))
        ub = 1 * np.ones((1, 2))
        return pypesto.Problem(objective=obj, lb=lb, ub=ub)
    else:
        raise ValueError("Unexpected input")


optimizers = [
    *[
        ('scipy', method)
        for method in [
            'Nelder-Mead',
            'Powell',
            'CG',
            'BFGS',
            'Newton-CG',
            'L-BFGS-B',
            'TNC',
            'COBYLA',
            'SLSQP',
            'trust-ncg',
            'trust-exact',
            'trust-krylov',
            'ls_trf',
            'ls_dogbox',
        ]
    ],
    # disabled: ,'trust-constr', 'ls_lm', 'dogleg'
    ('ipopt', ''),
    ('dlib', ''),
    ('pyswarm', ''),
    ('cmaes', ''),
    ('scipydiffevolopt', ''),
    ('pyswarms', ''),
    *[
        ('nlopt', method)
        for method in [
            nlopt.LD_VAR1,
            nlopt.LD_VAR2,
            nlopt.LD_TNEWTON_PRECOND_RESTART,
            nlopt.LD_TNEWTON_PRECOND,
            nlopt.LD_TNEWTON_RESTART,
            nlopt.LD_TNEWTON,
            nlopt.LD_LBFGS,
            nlopt.LD_SLSQP,
            nlopt.LD_CCSAQ,
            nlopt.LD_MMA,
            nlopt.LN_SBPLX,
            nlopt.LN_NELDERMEAD,
            nlopt.LN_PRAXIS,
            nlopt.LN_NEWUOA,
            nlopt.LN_NEWUOA_BOUND,
            nlopt.LN_BOBYQA,
            nlopt.LN_COBYLA,
            nlopt.GN_ESCH,
            nlopt.GN_ISRES,
            nlopt.GN_AGS,
            nlopt.GD_STOGO,
            nlopt.GD_STOGO_RAND,
            nlopt.G_MLSL,
            nlopt.G_MLSL_LDS,
            nlopt.GD_MLSL,
            nlopt.GD_MLSL_LDS,
            nlopt.GN_CRS2_LM,
            nlopt.GN_ORIG_DIRECT,
            nlopt.GN_ORIG_DIRECT_L,
            nlopt.GN_DIRECT,
            nlopt.GN_DIRECT_L,
            nlopt.GN_DIRECT_L_NOSCAL,
            nlopt.GN_DIRECT_L_RAND,
            nlopt.GN_DIRECT_L_RAND_NOSCAL,
            nlopt.AUGLAG,
            nlopt.AUGLAG_EQ,
        ]
    ],
    *[
        ('fides', solver)
        for solver in itt.product(
            [
                None,
                fides.BFGS(),
                fides.SR1(),
                fides.BB(),
                fides.BG(),
                fides.Broyden(0.5),
                fides.SSM(),
                fides.TSSM(),
                fides.HybridFixed(),
                fides.FX(),
                fides.GNSBFGS(),
            ],
            [
                fides.SubSpaceDim.TWO,
                fides.SubSpaceDim.FULL,
                fides.SubSpaceDim.STEIHAUG,
            ],
        )
    ],
]


@pytest.fixture(params=optimizers)
def optimizer(request):
    return request.param


@pytest.mark.flaky(reruns=5)
def test_optimization(problem, optimizer):
    """Test optimization using various optimizers and objective modes."""
    library, method = optimizer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_minimize(problem, library, method)


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
        lb,
        ub,
        lb_init=lb_init,
        ub_init=ub_init,
    )
    opt = get_optimizer(*optimizer)

    options = optimize.OptimizeOptions(allow_failed_starts=False)

    # check whether the optimizer is least squares
    if isinstance(optimizer[1], str) and re.match(r'(?i)^(ls_)', optimizer[1]):
        return

    if optimizer in [
        ('dlib', ''),
        ('pyswarm', ''),
        ('cmaes', ''),
        ('scipydiffevolopt', ''),
        ('pyswarms', ''),
        *[
            ('nlopt', method)
            for method in [
                nlopt.GN_ESCH,
                nlopt.GN_ISRES,
                nlopt.GN_AGS,
                nlopt.GD_STOGO,
                nlopt.GD_STOGO_RAND,
                nlopt.G_MLSL,
                nlopt.G_MLSL_LDS,
                nlopt.GD_MLSL,
                nlopt.GD_MLSL_LDS,
                nlopt.GN_CRS2_LM,
                nlopt.GN_ORIG_DIRECT,
                nlopt.GN_ORIG_DIRECT_L,
                nlopt.GN_DIRECT,
                nlopt.GN_DIRECT_L,
                nlopt.GN_DIRECT_L_NOSCAL,
                nlopt.GN_DIRECT_L_RAND,
                nlopt.GN_DIRECT_L_RAND_NOSCAL,
            ]
        ],
    ]:
        with pytest.raises(ValueError):
            optimize.minimize(
                problem=problem,
                optimizer=opt,
                n_starts=1,
                startpoint_method=pypesto.startpoint.uniform,
                options=options,
                filename=None,
                progress_bar=False,
            )
        return
    else:
        result = optimize.minimize(
            problem=problem,
            optimizer=opt,
            n_starts=1,
            startpoint_method=pypesto.startpoint.uniform,
            options=options,
            filename=None,
            progress_bar=False,
        )

    # check that ub/lb were reverted
    assert isinstance(result.optimize_result.list[0]['fval'], float)
    if optimizer not in [('scipy', 'ls_trf'), ('scipy', 'ls_dogbox')]:
        assert np.isfinite(result.optimize_result.list[0]['fval'])
        assert result.optimize_result.list[0]['x'] is not None
    # check that result is not in bounds, optimum is at (1,1), so you would
    # hope that any reasonable optimizer manage to finish with x < ub,
    # but I guess some are pretty terrible
    assert np.any(result.optimize_result.list[0]['x'] < lb_init) or np.any(
        result.optimize_result.list[0]['x'] > ub_init
    )


def get_optimizer(library, solver):
    """Constructs Optimizer given and optimization library and optimization
    solver specification"""
    options = {'maxiter': 100}

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
            options=options
        )
    elif library == 'pyswarms':
        optimizer = optimize.PyswarmsOptimizer(options=options)
    elif library == 'nlopt':
        optimizer = optimize.NLoptOptimizer(method=solver, options=options)
    elif library == 'fides':
        options[fides.Options.SUBSPACE_DIM] = solver[1]
        optimizer = optimize.FidesOptimizer(
            options=options, hessian_update=solver[0], verbose=40
        )
    else:
        raise ValueError(f"Optimizer not recognized: {library}")

    return optimizer


def check_minimize(problem, library, solver, allow_failed_starts=False):
    """Runs a single run of optimization according to the provided inputs
    and checks whether optimization yielded a solution."""
    optimizer = get_optimizer(library, solver)
    optimize_options = optimize.OptimizeOptions(
        allow_failed_starts=allow_failed_starts
    )

    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=1,
        startpoint_method=pypesto.startpoint.uniform,
        options=optimize_options,
        filename=None,
        progress_bar=False,
    )

    assert isinstance(result.optimize_result.list[0]['fval'], float)
    if (library, solver) not in [
        ('nlopt', nlopt.GD_STOGO_RAND)  # id 9, fails in 40% of cases
    ]:
        assert np.isfinite(result.optimize_result.list[0]['fval'])
        assert result.optimize_result.list[0]['x'] is not None


def test_trim_results(problem):
    """
    Test trimming of hess/sres from results
    """

    optimize_options = optimize.OptimizeOptions(
        report_hess=False, report_sres=False
    )
    prob = pypesto.Problem(
        objective=rosen_for_sensi(max_sensi_order=2)['obj'],
        lb=0 * np.ones((1, 2)),
        ub=1 * np.ones((1, 2)),
    )

    # hess
    optimizer = optimize.FidesOptimizer(verbose=40)
    result = optimize.minimize(
        problem=prob,
        optimizer=optimizer,
        n_starts=1,
        startpoint_method=pypesto.startpoint.uniform,
        options=optimize_options,
        filename=None,
        progress_bar=False,
    )
    assert result.optimize_result.list[0].hess is None

    # sres
    optimizer = optimize.ScipyOptimizer(method='ls_trf')
    result = optimize.minimize(
        problem=prob,
        optimizer=optimizer,
        n_starts=1,
        startpoint_method=pypesto.startpoint.uniform,
        options=optimize_options,
        filename=None,
        progress_bar=False,
    )
    assert result.optimize_result.list[0].sres is None


def test_mpipoolengine():
    """
    Test the MPIPoolEngine by calling an example script with mpiexec.
    """
    try:
        # get the path to this file:
        path = os.path.dirname(__file__)
        # run the example file.
        subprocess.check_call(  # noqa: S603,S607
            [
                'mpiexec',
                '-np',
                '2',
                'python',
                '-m',
                'mpi4py.futures',
                f'{path}/../../doc/example/example_MPIPool.py',
            ]
        )

        # read results
        result1 = read_result('temp_result.h5', problem=True, optimize=True)
        # set optimizer
        optimizer = optimize.FidesOptimizer(verbose=40)
        # initialize problem with x_guesses and objective
        objective = pypesto.Objective(
            fun=sp.optimize.rosen,
            grad=sp.optimize.rosen_der,
            hess=sp.optimize.rosen_hess,
        )
        x_guesses = np.array(
            [result1.optimize_result.list[i]['x0'] for i in range(2)]
        )
        problem = pypesto.Problem(
            objective=objective,
            ub=result1.problem.ub,
            lb=result1.problem.lb,
            x_guesses=x_guesses,
        )
        result2 = optimize.minimize(
            problem=problem,
            optimizer=optimizer,
            n_starts=2,
            engine=pypesto.engine.MultiProcessEngine(),
            filename=None,
            progress_bar=False,
        )

        for ix in range(2):
            assert_almost_equal(
                result1.optimize_result.list[ix]['x'],
                result2.optimize_result.list[ix]['x'],
                err_msg='The final parameter values '
                'do not agree for the engines.',
            )

    finally:
        if os.path.exists('temp_result.h5'):
            # delete data
            os.remove('temp_result.h5')


def test_history_beats_optimizer():
    """Test overwriting from history vs whatever the optimizer reports."""
    problem = CRProblem(
        x_guesses=np.array([0.25, 0.25]).reshape(1, -1)
    ).get_problem()

    max_fval = 10
    scipy_options = {"maxfun": max_fval}

    result_hist = optimize.minimize(
        problem=problem,
        optimizer=optimize.ScipyOptimizer(method="TNC", options=scipy_options),
        n_starts=1,
        options=optimize.OptimizeOptions(history_beats_optimizer=True),
        filename=None,
        progress_bar=False,
    )

    result_opt = optimize.minimize(
        problem=problem,
        optimizer=optimize.ScipyOptimizer(method="TNC", options=scipy_options),
        n_starts=1,
        options=optimize.OptimizeOptions(history_beats_optimizer=False),
        filename=None,
        progress_bar=False,
    )

    for result in (result_hist, result_opt):
        # number of function evaluations
        assert result.optimize_result.list[0]['n_fval'] <= max_fval + 1
        # optimal value in bounds
        assert np.all(problem.lb <= result.optimize_result.list[0]['x'])
        assert np.all(problem.ub >= result.optimize_result.list[0]['x'])
        # entries filled
        for key in ('fval', 'x', 'grad'):
            val = result.optimize_result.list[0][key]
            assert val is not None and np.all(np.isfinite(val))

    # TNC funnily reports the last value if not converged
    #  (this may break if their implementation is changed at some point ...)
    assert (
        result_hist.optimize_result.list[0]['fval']
        < result_opt.optimize_result.list[0]['fval']
    )
