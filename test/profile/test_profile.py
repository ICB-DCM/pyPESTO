"""
This is for testing profiling of the pypesto.Objective.
"""

import unittest
import warnings
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import pypesto
import pypesto.optimize as optimize
import pypesto.profile as profile
import pypesto.visualize as visualize
from pypesto import ObjectiveBase
from pypesto.profile.util import (
    precheck_profile_step_size,
    resolve_profile_step_sizes,
    resolve_profile_step_sizes_for_parameters,
)
from pypesto.profile.walk_along_profile import profile_multistart_optimize

from ..util import rosen_for_sensi
from ..visualize import close_fig


class ProfilerTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.objective: ObjectiveBase = rosen_for_sensi(
            max_sensi_order=2, integrated=True
        )["obj"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (
                cls.problem,
                cls.result,
                cls.optimizer,
            ) = create_optimization_results(cls.objective)

    @close_fig
    def test_default_profiling(self):
        # loop over  methods for creating new initial guesses
        method_list = [
            "fixed_step",
            "adaptive_step_order_0",
            "adaptive_step_order_1",
            "adaptive_step_regression",
        ]
        for i_run, method in enumerate(method_list):
            # run profiling
            result = profile.parameter_profile(
                problem=self.problem,
                result=self.result,
                optimizer=self.optimizer,
                next_guess_method=method,
                progress_bar=False,
            )

            # check result
            self.assertTrue(
                isinstance(
                    result.profile_result.list[i_run][0],
                    pypesto.ProfilerResult,
                )
            )
            self.assertEqual(len(result.profile_result.list), i_run + 1)
            self.assertEqual(len(result.profile_result.list[i_run]), 2)

            # check whether profiling needed maybe too many steps
            steps = result.profile_result.list[i_run][0]["ratio_path"].size
            if method == "adaptive_step_regression":
                self.assertTrue(
                    steps < 100,
                    "Profiling with regression based "
                    "proposal needed too many steps.",
                )
                self.assertTrue(
                    steps > 1,
                    "Profiling with regression based "
                    "proposal needed not enough steps.",
                )
            elif method == "adaptive_step_order_1":
                self.assertTrue(
                    steps < 100,
                    "Profiling with 1st order based "
                    "proposal needed too many steps.",
                )
                self.assertTrue(
                    steps > 1,
                    "Profiling with 1st order based "
                    "proposal needed not enough steps.",
                )
            elif method == "adaptive_step_order_0":
                self.assertTrue(
                    steps < 300,
                    "Profiling with 0th order based "
                    "proposal needed too many steps.",
                )
                self.assertTrue(
                    steps > 1,
                    "Profiling with 0th order based "
                    "proposal needed not enough steps.",
                )

            # standard plotting
            visualize.profiles(result, profile_list_ids=i_run)
            visualize.profile_cis(result, profile_list=i_run)

    def test_engine_profiling(self):
        # loop over all possible engines
        # engine=None will be used for comparison
        engines = [
            None,
            pypesto.engine.SingleCoreEngine(),
            pypesto.engine.MultiProcessEngine(),
            pypesto.engine.MultiThreadEngine(),
        ]
        expected_warns = [
            pytest.warns(UserWarning, match="fun and hess as one func"),
            pytest.warns(UserWarning, match="fun and hess as one func"),
            warnings.catch_warnings(),  # No warnings
            warnings.catch_warnings(),  # No warnings
        ]
        for engine, expected_warn in zip(engines, expected_warns, strict=True):
            # run profiling, profile results get appended
            # in self.result.profile_result
            with expected_warn:
                profile.parameter_profile(
                    problem=self.problem,
                    result=self.result,
                    optimizer=self.optimizer,
                    next_guess_method="fixed_step",
                    engine=engine,
                    progress_bar=False,
                )

        # check results
        for count, _engine in enumerate(engines[1:]):
            for j in range(len(self.result.profile_result.list[0])):
                assert_almost_equal(
                    self.result.profile_result.list[0][j]["x_path"],
                    self.result.profile_result.list[count][j]["x_path"],
                    err_msg="The values of the profiles for"
                    " the different engines do not match",
                )

    def test_selected_profiling(self):
        # create options in order to ensure a short computation time
        options = profile.ProfileOptions(
            default_step_size_absolute=0.02,
            min_step_size_absolute=0.005,
            max_step_size_absolute=1.0,
            step_size_factor=1.5,
            delta_ratio_max=0.2,
            ratio_min=0.3,
            reg_points=5,
            reg_order=2,
        )

        # 1st run of profiling, computing just one out of two profiles
        result = profile.parameter_profile(
            problem=self.problem,
            result=self.result,
            optimizer=self.optimizer,
            profile_index=np.array([1]),
            next_guess_method="fixed_step",
            result_index=1,
            profile_options=options,
            progress_bar=False,
        )

        self.assertIsInstance(
            result.profile_result.list[0][1], pypesto.ProfilerResult
        )
        self.assertIsNone(result.profile_result.list[0][0])

        # 2nd run of profiling, appending to an existing list of profiles
        # using another algorithm and another optimum
        result = profile.parameter_profile(
            problem=self.problem,
            result=result,
            optimizer=self.optimizer,
            profile_index=np.array([0]),
            result_index=2,
            profile_list=0,
            profile_options=options,
            progress_bar=False,
        )

        self.assertIsInstance(
            result.profile_result.list[0][0], pypesto.ProfilerResult
        )

        # 3rd run of profiling, opening a new list, using the default algorithm
        result = profile.parameter_profile(
            problem=self.problem,
            result=result,
            optimizer=self.optimizer,
            next_guess_method="fixed_step",
            profile_index=np.array([0]),
            profile_options=options,
            progress_bar=False,
        )
        # check result
        self.assertIsInstance(
            result.profile_result.list[1][0], pypesto.ProfilerResult
        )
        self.assertIsNone(result.profile_result.list[1][1])

    def test_extending_profiles(self):
        # run profiling
        result = profile.parameter_profile(
            problem=self.problem,
            result=self.result,
            optimizer=self.optimizer,
            next_guess_method="fixed_step",
            progress_bar=False,
        )

        # set new bounds (knowing that one parameter stopped at the bounds
        self.problem.lb_full = -4 * np.ones(2)
        self.problem.ub_full = 4 * np.ones(2)

        # re-run profiling using new bounds
        result = profile.parameter_profile(
            problem=self.problem,
            result=result,
            optimizer=self.optimizer,
            next_guess_method="fixed_step",
            profile_index=np.array([1]),
            profile_list=0,
            progress_bar=False,
        )
        # check result
        self.assertTrue(
            isinstance(
                result.profile_result.list[0][0], pypesto.ProfilerResult
            )
        )
        self.assertTrue(
            isinstance(
                result.profile_result.list[0][1], pypesto.ProfilerResult
            )
        )

    def test_approximate_profiles(self):
        """Test for the approximate profile function."""
        n_steps = 50
        assert self.result.optimize_result.list[0].hess is None
        result = profile.approximate_parameter_profile(
            problem=self.problem,
            result=self.result,
            profile_index=[1],
            n_steps=n_steps,
        )
        profile_list = result.profile_result.list[-1]
        assert profile_list[0] is None
        assert isinstance(profile_list[1], pypesto.ProfilerResult)
        assert np.isclose(profile_list[1].ratio_path.max(), 1)
        assert len(profile_list[1].ratio_path) == n_steps
        assert profile_list[1].x_path.shape == (2, n_steps)

        # with pre-defined hessian
        result = deepcopy(self.result)
        result.optimize_result.list[0].hess = np.array([[2, 0], [0, 1]])
        profile.approximate_parameter_profile(
            problem=self.problem,
            result=result,
            profile_index=[1],
            n_steps=n_steps,
        )


# dont make this a class method such that we dont optimize twice
def test_profile_with_history():
    objective = rosen_for_sensi(max_sensi_order=2, integrated=False)["obj"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (problem, result, optimizer) = create_optimization_results(
            objective, dim_full=5
        )

    profile_options = profile.ProfileOptions(
        min_step_size_absolute=0.0005,
        delta_ratio_max=0.05,
        default_step_size_absolute=0.005,
        ratio_min=0.03,
    )

    problem.fix_parameters(
        [0, 3],
        [
            result.optimize_result.list[0].x[0],
            result.optimize_result.list[0].x[3],
        ],
    )
    problem.objective.history = pypesto.MemoryHistory({"trace_record": True})
    profile.parameter_profile(
        problem=problem,
        result=result,
        optimizer=optimizer,
        profile_index=np.array([0, 2, 4]),
        result_index=0,
        profile_options=profile_options,
        progress_bar=False,
    )


@close_fig
def test_profile_with_fixed_parameters():
    """Test using profiles with fixed parameters."""
    obj = rosen_for_sensi(max_sensi_order=1)["obj"]

    lb = -2 * np.ones(5)
    ub = 2 * np.ones(5)
    problem = pypesto.Problem(
        objective=obj,
        lb=lb,
        ub=ub,
        x_fixed_vals=[0.5, -1.8],
        x_fixed_indices=[0, 3],
    )

    optimizer = optimize.ScipyOptimizer(options={"maxiter": 50})
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=2,
        progress_bar=False,
    )

    for i_method, next_guess_method in enumerate(
        [
            "fixed_step",
            "adaptive_step_order_0",
            "adaptive_step_order_1",
            "adaptive_step_regression",
        ]
    ):
        print(next_guess_method)
        profile.parameter_profile(
            problem=problem,
            result=result,
            optimizer=optimizer,
            next_guess_method=next_guess_method,
            progress_bar=False,
        )

        # standard plotting
        axes = visualize.profiles(result, profile_list_ids=i_method)
        assert len(axes) == 3
        visualize.profile_cis(result, profile_list=i_method)

    # test profiling with all parameters fixed but one
    problem.fix_parameters([2, 3, 4], result.optimize_result.list[0]["x"][2:5])
    resolved_steps_by_par = resolve_profile_step_sizes_for_parameters(
        problem, problem.x_free_indices, profile.ProfileOptions()
    )
    assert set(resolved_steps_by_par) == set(problem.x_free_indices)
    profile.parameter_profile(
        problem=problem,
        result=result,
        optimizer=optimizer,
        next_guess_method="adaptive_step_regression",
        progress_bar=False,
    )


def create_optimization_results(objective, dim_full=2):
    # create optimizer, pypesto problem and options
    options = {"maxiter": 200}
    optimizer = optimize.ScipyOptimizer(method="l-bfgs-b", options=options)

    lb = -2 * np.ones(dim_full)
    ub = 2 * np.ones(dim_full)
    problem = pypesto.Problem(objective, lb, ub)

    optimize_options = optimize.OptimizeOptions(allow_failed_starts=True)

    # run optimization
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=5,
        startpoint_method=pypesto.startpoint.uniform,
        options=optimize_options,
        progress_bar=False,
    )

    return problem, result, optimizer


def test_chi2_quantile_to_ratio():
    """Tests the chi2 quantile to ratio convenience function."""
    ratio = profile.chi2_quantile_to_ratio()
    assert np.isclose(ratio, 0.1465)


def test_approximate_ci():
    xs = np.array([-3, -1, 1, 3, 5, 7, 9])

    ratios = np.array([0.2, 0.3, 1, 0.27, 0.15, 0.15, 0.1])

    lb, ub = profile.calculate_approximate_ci(
        xs=xs, ratios=ratios, confidence_ratio=0.27
    )

    # correct interpolation
    assert np.isclose(lb, -3 + (-1 - (-3)) * 0.7)

    # exact pick
    assert np.isclose(ub, 3)

    lb, ub = profile.calculate_approximate_ci(
        xs=xs, ratios=ratios, confidence_ratio=0.15
    )

    # double value
    assert np.isclose(ub, 7)

    lb, ub = profile.calculate_approximate_ci(
        xs=xs, ratios=ratios, confidence_ratio=0.1
    )

    # bound value
    assert np.isclose(lb, -3)
    assert np.isclose(ub, 9)


def test_options_valid():
    """Test ProfileOptions validity checks."""
    # default settings are valid
    profile.ProfileOptions()
    profile.ProfileOptions(
        min_step_size_relative=0.0025,
        default_step_size_relative=0.005,
        max_step_size_relative=0.02,
    )

    # try to set invalid values
    with pytest.raises(ValueError):
        profile.ProfileOptions(default_step_size_absolute=-1)
    with pytest.raises(ValueError):
        profile.ProfileOptions(default_step_size_relative=-0.01)
    with pytest.warns(DeprecationWarning, match="`default_step_size`"):
        options = profile.ProfileOptions(default_step_size=0.05)
    assert options.default_step_size_absolute == 0.05
    # the deprecated argument overrides the new one
    with pytest.warns(DeprecationWarning, match="`default_step_size`"):
        options = profile.ProfileOptions(
            default_step_size=0.01,
            default_step_size_absolute=0.03,
        )
    assert options.default_step_size_absolute == 0.01
    # the deprecated attribute is still readable
    with pytest.warns(DeprecationWarning, match="`default_step_size`"):
        assert options.default_step_size == 0.01
    for kwargs in (
        {
            "default_step_size_absolute": 1,
            "min_step_size_absolute": 2,
        },
        {
            "default_step_size_absolute": 2,
            "min_step_size_absolute": 1,
            "max_step_size_absolute": 1,
        },
        {
            "min_step_size_relative": 0.006,
            "default_step_size_relative": 0.005,
        },
        {
            "default_step_size_relative": 0.03,
            "max_step_size_relative": 0.02,
        },
        {"correlation_threshold": -0.1},
        {"correlation_threshold": 1.1},
        {"profile_n_starts": 0},
        {"profile_sampling_sigma": 0},
        {
            "default_step_size_absolute": 0.0,
            "default_step_size_relative": 0.0,
        },
        {"step_size_precheck_mode": "invalid"},
    ):
        with pytest.raises(ValueError):
            profile.ProfileOptions(**kwargs)


@pytest.mark.parametrize(
    (
        "scale",
        "lb",
        "ub",
        "profile_options",
        "expected_min",
        "expected_default",
        "expected_max",
        "expected_mode",
    ),
    [
        ("lin", 0.0, 100.0, None, 0.5, 1.0, 4.0, "relative"),
        ("lin", 0.0, 1.0, None, 0.01, 0.02, 0.2, "absolute"),
        ("log10", -6.0, 6.0, None, 0.06, 0.12, 0.48, "relative"),
        (
            "lin",
            0.0,
            100.0,
            profile.ProfileOptions(
                min_step_size_absolute=0.1,
                default_step_size_absolute=0.5,
                max_step_size_absolute=10.0,
                min_step_size_relative=0.002,
                default_step_size_relative=0.005,
                max_step_size_relative=0.006,
            ),
            0.2,
            0.5,
            0.6,
            "relative",
        ),
    ],
)
def test_resolve_profile_step_sizes(
    scale,
    lb,
    ub,
    profile_options,
    expected_min,
    expected_default,
    expected_max,
    expected_mode,
):
    """Resolved step sizes should pick one family on the optimization scale."""
    problem = pypesto.Problem(
        objective=pypesto.Objective(fun=lambda x: np.sum(x**2)),
        lb=np.array([lb]),
        ub=np.array([ub]),
        x_scales=[scale],
        x_names=["x0"],
    )
    resolved_steps = resolve_profile_step_sizes(
        problem,
        0,
        profile_options or profile.ProfileOptions(),
    )

    assert np.isclose(resolved_steps.min_step_size, expected_min)
    assert np.isclose(resolved_steps.default_step_size, expected_default)
    assert np.isclose(resolved_steps.max_step_size, expected_max)
    assert resolved_steps.mode == expected_mode
    assert np.isclose(resolved_steps.span, ub - lb)
    assert (
        resolve_profile_step_sizes_for_parameters(
            problem,
            [0],
            profile_options or profile.ProfileOptions(),
        )[0]
        == resolved_steps
    )


@pytest.mark.parametrize(
    ("mode", "expect_warning", "expect_raise"),
    [
        ("off", False, False),
        ("warn", True, False),
        ("raise", False, True),
    ],
)
def test_profile_step_size_precheck_modes(mode, expect_warning, expect_raise):
    """Precheck modes should suppress, warn, or raise on large spans."""
    problem = pypesto.Problem(
        objective=pypesto.Objective(fun=lambda x: np.sum(x**2)),
        lb=np.array([-5.0]),
        ub=np.array([15.0]),
        x_scales=["log10"],
        x_names=["x0"],
    )
    current_profile = pypesto.ProfilerResult(
        x_path=np.array([[0.0]]),
        fval_path=np.array([0.0]),
        ratio_path=np.array([1.0]),
    )
    profile_options = profile.ProfileOptions(
        min_step_size_relative=0.0005,
        default_step_size_relative=0.001,
        max_step_size_relative=0.01,
        step_size_precheck_mode=mode,
        whole_path=True,
    )
    resolved_steps = resolve_profile_step_sizes(problem, 0, profile_options)

    if expect_raise:
        with pytest.raises(ValueError, match="may require many steps"):
            precheck_profile_step_size(
                current_profile=current_profile,
                problem=problem,
                i_par=0,
                par_direction=1,
                options=profile_options,
                resolved_steps=resolved_steps,
            )
        return

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        precheck_profile_step_size(
            current_profile=current_profile,
            problem=problem,
            i_par=0,
            par_direction=1,
            options=profile_options,
            resolved_steps=resolved_steps,
        )

    precheck_warnings = [
        warning
        for warning in caught
        if "may require many steps" in str(warning.message)
    ]
    if expect_warning:
        assert precheck_warnings
        message = str(precheck_warnings[0].message)
        assert "default step size" in message
        assert "minimum step size" in message
    else:
        assert not precheck_warnings


def test_profile_multistart_optimize_uses_best_start(monkeypatch):
    """Multi-start profiling should tolerate failed starts and keep the best finite result."""

    class DummyOptimizer:
        def __init__(self):
            self.calls = []

        def minimize(
            self,
            problem,
            x0=None,
            id=None,
            history_options=None,
            optimize_options=None,
        ):
            del problem, history_options
            self.calls.append(np.array(x0, copy=True))
            if np.isclose(x0[0], 0.5):
                if optimize_options.allow_failed_starts:
                    # Real pyPESTO optimizers back-fill failed tolerated
                    # starts from history, which leaves them non-finite if no
                    # useful point was recorded before the exception.
                    return pypesto.OptimizerResult(
                        id=id,
                        x0=np.array(x0, copy=True),
                        fval=np.inf,
                        exitflag=-1,
                        message="sampled start failed",
                    )
                raise RuntimeError("sampled start failed")
            return pypesto.OptimizerResult(
                id=id,
                x=np.array(x0, copy=True),
                fval=float(np.sum(x0**2)),
            )

    problem = pypesto.Problem(
        objective=pypesto.Objective(fun=lambda x: x[0] ** 2),
        lb=np.array([-1.0]),
        ub=np.array([1.0]),
    )
    startpoint = np.array([0.8])
    options = profile.ProfileOptions(profile_n_starts=3)

    monkeypatch.setattr(
        np.random,
        "normal",
        lambda loc, scale, size: np.array([[0.5], [0.1]]),
    )

    optimizer = DummyOptimizer()
    result = profile_multistart_optimize(
        optimizer=optimizer,
        problem=problem,
        startpoint=startpoint,
        options=options,
    )

    assert len(optimizer.calls) == options.profile_n_starts
    assert np.allclose(optimizer.calls[-1], startpoint)
    assert np.allclose(result.x, np.array([0.1]))
    assert np.isclose(result.fval, 0.01)


@pytest.mark.parametrize(
    "lb,ub",
    [(6 * np.ones(5), 10 * np.ones(5)), (-4 * np.ones(5), 1 * np.ones(5))],
)
def test_gh1165(lb, ub):
    """Regression test for https://github.com/ICB-DCM/pyPESTO/issues/1165

    Check profiles with non-symmetric bounds and whole_path=True span the full parameter domain.
    """
    obj = rosen_for_sensi(max_sensi_order=1)["obj"]

    problem = pypesto.Problem(
        objective=obj,
        lb=lb,
        ub=ub,
    )

    optimizer = optimize.ScipyOptimizer(options={"maxiter": 10})
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=2,
        progress_bar=False,
    )
    # just any parameter
    par_idx = 1
    profile.parameter_profile(
        problem=problem,
        result=result,
        optimizer=optimizer,
        next_guess_method="fixed_step",
        profile_index=[par_idx],
        progress_bar=False,
        profile_options=profile.ProfileOptions(
            min_step_size_absolute=0.1,
            max_step_size_absolute=1.0,
            delta_ratio_max=0.05,
            default_step_size_absolute=0.5,
            ratio_min=0.01,
            whole_path=True,
        ),
    )
    # parameter value of the profiled parameter
    x_path = result.profile_result.list[0][par_idx]["x_path"][par_idx, :]
    # ensure we cover lb..ub
    assert x_path[0] == lb[par_idx], (x_path.min(), lb[par_idx])
    assert x_path[-1] == ub[par_idx], (x_path.max(), ub[par_idx])
