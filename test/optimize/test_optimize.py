"""
This is for testing optimization of the pypesto.Objective.
"""

import itertools as itt
import logging
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
from pypesto.optimize.ess import (
    ESSOptimizer,
    SacessFidesFactory,
    SacessOptimizer,
    get_default_ess_options,
)
from pypesto.optimize.util import assign_ids
from pypesto.store import read_result

from ..base.test_x_fixed import create_problem
from ..util import CRProblem, rosen_for_sensi


@pytest.fixture(params=["cr", "rosen-integrated", "rosen-separated"])
def problem(request) -> pypesto.Problem:
    if request.param == "cr":
        return CRProblem().get_problem()
    elif "rosen" in request.param:
        integrated = "integrated" in request.param
        obj = rosen_for_sensi(max_sensi_order=2, integrated=integrated)["obj"]
        lb = 0 * np.ones((1, 2))
        ub = 1 * np.ones((1, 2))
        return pypesto.Problem(objective=obj, lb=lb, ub=ub)
    else:
        raise ValueError("Unexpected input")


optimizers = [
    *[
        ("scipy", method)
        for method in [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "dogleg",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
            "ls_trf",
            "ls_dogbox",
        ]
    ],
    # disabled: 'ls_lm' (ValueError when passing bounds)
    ("ipopt", ""),
    ("dlib", ""),
    ("pyswarm", ""),
    ("cma", ""),
    ("scipydiffevolopt", ""),
    ("pyswarms", ""),
    *[
        ("nlopt", method)
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
        ("fides", solver)
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


@pytest.fixture(
    params=optimizers,
    ids=[
        f"{i}-{o[0]}{'-' + str(o[1]) if isinstance(o[1], str) and o[1] else ''}"
        for i, o in enumerate(optimizers)
    ],
)
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
        rosen_for_sensi(max_sensi_order=2)["obj"],
        lb,
        ub,
        lb_init=lb_init,
        ub_init=ub_init,
    )
    opt = get_optimizer(*optimizer)

    options = optimize.OptimizeOptions(allow_failed_starts=False)

    # check whether the optimizer is least squares
    if isinstance(optimizer[1], str) and re.match(r"(?i)^(ls_)", optimizer[1]):
        return

    if optimizer in [
        ("dlib", ""),
        ("pyswarm", ""),
        ("cma", ""),
        ("scipydiffevolopt", ""),
        ("pyswarms", ""),
        *[
            ("nlopt", method)
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
                options=options,
                progress_bar=False,
            )
        return
    else:
        result = optimize.minimize(
            problem=problem,
            optimizer=opt,
            n_starts=1,
            options=options,
            progress_bar=False,
        )

    # check that ub/lb were reverted
    assert isinstance(result.optimize_result.list[0]["fval"], float)
    if optimizer not in [("scipy", "ls_trf"), ("scipy", "ls_dogbox")]:
        assert np.isfinite(result.optimize_result.list[0]["fval"])
        assert result.optimize_result.list[0]["x"] is not None
    # check that result is not in bounds, optimum is at (1,1), so you would
    # hope that any reasonable optimizer manage to finish with x < ub,
    # but I guess some are pretty terrible
    assert np.any(result.optimize_result.list[0]["x"] < lb_init) or np.any(
        result.optimize_result.list[0]["x"] > ub_init
    )


def get_optimizer(library, solver):
    """Constructs Optimizer given and optimization library and optimization
    solver specification"""
    options = {"maxiter": 100}

    if library == "scipy":
        if solver == "TNC" or solver.startswith("ls_"):
            options["maxfun"] = options.pop("maxiter")
        optimizer = optimize.ScipyOptimizer(method=solver, options=options)
    elif library == "ipopt":
        optimizer = optimize.IpoptOptimizer()
    elif library == "dlib":
        optimizer = optimize.DlibOptimizer(options=options)
    elif library == "pyswarm":
        optimizer = optimize.PyswarmOptimizer(options=options)
    elif library == "cma":
        optimizer = optimize.CmaOptimizer(options=options)
    elif library == "scipydiffevolopt":
        optimizer = optimize.ScipyDifferentialEvolutionOptimizer(
            options=options
        )
    elif library == "pyswarms":
        optimizer = optimize.PyswarmsOptimizer(options=options)
    elif library == "nlopt":
        optimizer = optimize.NLoptOptimizer(method=solver, options=options)
    elif library == "fides":
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
        options=optimize_options,
        progress_bar=False,
    )

    assert isinstance(result.optimize_result.list[0]["fval"], float)
    if (library, solver) not in [
        ("nlopt", nlopt.GD_STOGO_RAND)  # id 9, fails in 40% of cases
    ]:
        assert np.isfinite(result.optimize_result.list[0]["fval"])
        assert result.optimize_result.list[0]["x"] is not None


def test_trim_results(problem):
    """
    Test trimming of hess/sres from results
    """

    optimize_options = optimize.OptimizeOptions(
        report_hess=False, report_sres=False
    )
    prob = pypesto.Problem(
        objective=rosen_for_sensi(max_sensi_order=2)["obj"],
        lb=0 * np.ones((1, 2)),
        ub=1 * np.ones((1, 2)),
    )

    # hess
    optimizer = optimize.FidesOptimizer(verbose=40)
    result = optimize.minimize(
        problem=prob,
        optimizer=optimizer,
        n_starts=1,
        options=optimize_options,
        progress_bar=False,
    )
    assert result.optimize_result.list[0].hess is None

    # sres
    optimizer = optimize.ScipyOptimizer(method="ls_trf")
    result = optimize.minimize(
        problem=prob,
        optimizer=optimizer,
        n_starts=1,
        options=optimize_options,
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
        subprocess.check_call(
            [  # noqa: S603,S607
                "mpiexec",
                "--oversubscribe",
                "-np",
                "2",
                "python",
                "-m",
                "mpi4py.futures",
                f"{path}/../../doc/example/example_MPIPool.py",
            ]
        )

        # read results
        result1 = read_result("temp_result.h5", problem=True, optimize=True)
        # set optimizer
        optimizer = optimize.FidesOptimizer(verbose=40)
        # initialize problem with x_guesses and objective
        objective = pypesto.Objective(
            fun=sp.optimize.rosen,
            grad=sp.optimize.rosen_der,
            hess=sp.optimize.rosen_hess,
        )
        x_guesses = np.array(
            [result1.optimize_result.list[i]["x0"] for i in range(2)]
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
            progress_bar=False,
        )

        for ix in range(2):
            assert_almost_equal(
                result1.optimize_result.list[ix]["x"],
                result2.optimize_result.list[ix]["x"],
                err_msg="The final parameter values "
                "do not agree for the engines.",
            )

    finally:
        if os.path.exists("temp_result.h5"):
            # delete data
            os.remove("temp_result.h5")


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
        progress_bar=False,
    )

    result_opt = optimize.minimize(
        problem=problem,
        optimizer=optimize.ScipyOptimizer(method="TNC", options=scipy_options),
        n_starts=1,
        options=optimize.OptimizeOptions(history_beats_optimizer=False),
        progress_bar=False,
    )

    for result in (result_hist, result_opt):
        # number of function evaluations
        assert result.optimize_result.list[0]["n_fval"] <= max_fval + 1
        # optimal value in bounds
        assert np.all(problem.lb <= result.optimize_result.list[0]["x"])
        assert np.all(problem.ub >= result.optimize_result.list[0]["x"])
        # entries filled
        for key in ("fval", "x", "grad"):
            val = result.optimize_result.list[0][key]
            assert val is not None and np.all(np.isfinite(val))

    # TNC funnily reports the last value if not converged
    #  (this may break if their implementation is changed at some point ...)
    assert (
        result_hist.optimize_result.list[0]["fval"]
        < result_opt.optimize_result.list[0]["fval"]
    )


@pytest.mark.filterwarnings(
    "ignore:Passing `startpoint_method` directly is deprecated.*:DeprecationWarning"
)
@pytest.mark.parametrize("ess_type", ["ess", "sacess"])
@pytest.mark.parametrize(
    "local_optimizer",
    [None, optimize.FidesOptimizer(), SacessFidesFactory()],
)
@pytest.mark.flaky(reruns=3)
def test_ess(problem, local_optimizer, ess_type, request):
    if ess_type == "ess":
        ess = ESSOptimizer(
            dim_refset=10,
            max_iter=20,
            local_optimizer=local_optimizer,
            local_n1=15,
            local_n2=5,
            n_threads=2,
            balance=0.5,
        )
    elif ess_type == "sacess":
        if (
            "cr" in request.node.callspec.id
            or "integrated" in request.node.callspec.id
        ):
            # Not pickleable - incompatible with CESS
            pytest.skip()
        # SACESS with 12 processes
        #  We use a higher number than reasonable to be more likely to trigger
        #  any potential race conditions (gh-1204)
        ess_init_args = get_default_ess_options(
            num_workers=12, dim=problem.dim
        )
        for x in ess_init_args:
            x["local_optimizer"] = local_optimizer
        ess = SacessOptimizer(
            max_walltime_s=1,
            sacess_loglevel=logging.DEBUG,
            ess_loglevel=logging.WARNING,
            ess_init_args=ess_init_args,
        )
    else:
        raise ValueError(f"Unsupported ESS type {ess_type}.")

    res = ess.minimize(
        problem=problem,
    )
    print("ESS result: ", res.summary())

    # best values roughly: cr: 4.701; rosen 7.592e-10
    if "rosen" in request.node.callspec.id:
        if local_optimizer:
            assert res.optimize_result[0].fval < 1e-4
        assert res.optimize_result[0].fval < 1
    elif "cr" in request.node.callspec.id:
        if local_optimizer:
            assert res.optimize_result[0].fval < 5
        assert res.optimize_result[0].fval < 20
    else:
        raise AssertionError()


def test_ess_multiprocess(problem, request):
    if (
        "cr" in request.node.callspec.id
        or "integrated" in request.node.callspec.id
    ):
        # Not pickleable - incompatible with CESS
        pytest.skip()

    from fides.constants import Options as FidesOptions

    from pypesto.optimize.ess import ESSOptimizer, FunctionEvaluatorMP, RefSet

    ess = ESSOptimizer(
        max_iter=20,
        # also test passing a callable as local_optimizer
        local_optimizer=lambda max_walltime_s,
        **kwargs: optimize.FidesOptimizer(
            options={FidesOptions.MAXTIME: max_walltime_s}
        ),
    )
    refset = RefSet(
        dim=10,
        evaluator=FunctionEvaluatorMP(
            problem=problem,
            startpoint_method=pypesto.startpoint.UniformStartpoints(),
            n_procs=4,
        ),
    )
    refset.initialize_random(10 * refset.dim)
    res = ess.minimize(
        refset=refset,
    )
    print("ESS result: ", res.summary())


def test_scipy_integrated_grad():
    integrated = True
    obj = rosen_for_sensi(max_sensi_order=2, integrated=integrated)["obj"]
    lb = 0 * np.ones((1, 2))
    ub = 1 * np.ones((1, 2))
    x_guesses = [[0.5, 0.5]]
    problem = pypesto.Problem(objective=obj, lb=lb, ub=ub, x_guesses=x_guesses)
    optimizer = optimize.ScipyOptimizer(options={"maxiter": 10})
    optimize_options = optimize.OptimizeOptions(allow_failed_starts=False)
    history_options = pypesto.HistoryOptions(trace_record=True)
    with pytest.warns(UserWarning, match="fun and hess as one func"):
        result = optimize.minimize(
            problem=problem,
            optimizer=optimizer,
            n_starts=1,
            options=optimize_options,
            history_options=history_options,
            progress_bar=False,
        )
    assert (
        len(result.optimize_result.history[0].get_fval_trace())
        == result.optimize_result.history[0].n_fval
    )


def test_ipopt_approx_grad():
    integrated = False
    obj = rosen_for_sensi(max_sensi_order=0, integrated=integrated)["obj"]
    lb = 0 * np.ones((1, 2))
    ub = 1 * np.ones((1, 2))
    x_guesses = [[0.5, 0.5]]
    problem = pypesto.Problem(objective=obj, lb=lb, ub=ub, x_guesses=x_guesses)
    optimizer = optimize.IpoptOptimizer(
        options={"maxiter": 10, "approx_grad": True}
    )
    optimize_options = optimize.OptimizeOptions(allow_failed_starts=False)
    history_options = pypesto.HistoryOptions(trace_record=True)
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=1,
        options=optimize_options,
        history_options=history_options,
        progress_bar=False,
    )
    obj2 = rosen_for_sensi(max_sensi_order=1, integrated=integrated)["obj"]
    problem2 = pypesto.Problem(
        objective=obj2, lb=lb, ub=ub, x_guesses=x_guesses
    )
    optimizer2 = optimize.IpoptOptimizer(options={"maxiter": 10})
    result2 = optimize.minimize(
        problem=problem2,
        optimizer=optimizer2,
        n_starts=1,
        options=optimize_options,
        history_options=history_options,
        progress_bar=False,
    )
    np.testing.assert_array_almost_equal(
        result.optimize_result[0].x, result2.optimize_result[0].x, decimal=4
    )


def test_correct_startpoint_usage(optimizer):
    """
    Test that the startpoint is correctly used in all optimizers.
    """
    # cma supports x0, but samples from this initial guess, therefore return
    if optimizer == ("cma", ""):
        return

    opt = get_optimizer(*optimizer)
    # return if the optimizer knowingly does not support x_guesses
    if not opt.check_x0_support():
        return

    # define a problem with an x_guess
    problem = CRProblem(x_guesses=[np.array([0.1, 0.1])]).get_problem()

    # run optimization
    result = optimize.minimize(
        problem=problem,
        optimizer=opt,
        n_starts=1,
        progress_bar=False,
        history_options=pypesto.HistoryOptions(trace_record=True),
    )
    # check that the startpoint was used
    assert problem.x_guesses[0] == pytest.approx(
        result.optimize_result[0].history.get_x_trace(0)
    )


def test_summary(caplog):
    """Test the result summary."""
    problem = create_problem()
    optimizer = pypesto.optimize.ScipyOptimizer()
    n_starts = 5
    result = pypesto.optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=n_starts,
        progress_bar=False,
    )

    # test that both full and reduced summary are available
    assert isinstance(result.summary(full=False), str)
    assert isinstance(result.summary(full=True), str)  # creates warning

    # as we have fixed parameters, the string of full should be longer
    assert len(result.summary(full=False)) < len(result.summary(full=True))

    # test that a warning is correctly printed.
    result.optimize_result[0].free_indices = None
    result.optimize_result[0].summary(full=True)
    expected_warning = (
        "There is no information about fixed parameters, "
        "run update_to_full with the corresponding problem first."
    )
    assert expected_warning in [r.message for r in caplog.records]


def test_assign_ids():
    n_starts = 5
    test_ids = [str(i) for i in range(n_starts)]
    result = pypesto.Result()
    result.optimize_result.id = test_ids

    ids = assign_ids(n_starts=n_starts, ids=None, result=result)
    assert ids == [str(i) for i in range(n_starts, n_starts * 2)]
