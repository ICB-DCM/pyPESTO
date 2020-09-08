"""
This is for testing profiling of the pypesto.Objective.
"""

import numpy as np
import unittest
import test.test_objective as test_objective
from copy import deepcopy
import warnings

import pypesto
import pypesto.optimize as optimize
import pypesto.profile as profile
import pypesto.visualize as visualize
from pypesto import ObjectiveBase
from .visualize import close_fig


class ProfilerTest(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.objective: ObjectiveBase = test_objective.rosen_for_sensi(
            max_sensi_order=2, integrated=True
        )['obj']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (cls.problem, cls.result, cls.optimizer) = \
                create_optimization_results(cls.objective)

    @close_fig
    def test_default_profiling(self):
        # loop over  methods for creating new initial guesses
        method_list = ['fixed_step', 'adaptive_step_order_0',
                       'adaptive_step_order_1', 'adaptive_step_regression']
        for i_run, method in enumerate(method_list):
            # run profiling
            result = profile.parameter_profile(
                problem=self.problem,
                result=self.result,
                optimizer=self.optimizer,
                next_guess_method=method)

            # check result
            self.assertTrue(
                isinstance(result.profile_result.list[i_run][0],
                           profile.ProfilerResult))
            self.assertEqual(len(result.profile_result.list), i_run+1)
            self.assertEqual(len(result.profile_result.list[i_run]), 2)

            # check whether profiling needed maybe too many steps
            steps = result.profile_result.list[i_run][0]['ratio_path'].size
            if method == 'adaptive_step_regression':
                self.assertTrue(steps < 20, 'Profiling with regression based '
                                            'proposal needed too many steps.')
                self.assertTrue(steps > 1, 'Profiling with regression based '
                                           'proposal needed not enough steps.')
            elif method == 'adaptive_step_order_1':
                self.assertTrue(steps < 25, 'Profiling with 1st order based '
                                            'proposal needed too many steps.')
                self.assertTrue(steps > 1, 'Profiling with 1st order based '
                                           'proposal needed not enough steps.')
            elif method == 'adaptive_step_order_0':
                self.assertTrue(steps < 100, 'Profiling with 0th order based '
                                             'proposal needed too many steps.')
                self.assertTrue(steps > 1, 'Profiling with 0th order based '
                                           'proposal needed not enough steps.')

            # standard plotting
            visualize.profiles(result, profile_list_ids=i_run)
            visualize.profile_cis(result, profile_list=i_run)

    def test_selected_profiling(self):
        # create options in order to ensure a short computation time
        options = profile.ProfileOptions(
            default_step_size=0.02,
            min_step_size=0.005,
            max_step_size=1.,
            step_size_factor=1.5,
            delta_ratio_max=0.2,
            ratio_min=0.3,
            reg_points=5,
            reg_order=2)

        # 1st run of profiling, computing just one out of two profiles
        result = profile.parameter_profile(
            problem=self.problem,
            result=self.result,
            optimizer=self.optimizer,
            profile_index=np.array([1]),
            next_guess_method='fixed_step',
            result_index=1,
            profile_options=options)

        self.assertIsInstance(result.profile_result.list[0][1],
                              profile.ProfilerResult)
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
            profile_options=options)

        self.assertIsInstance(result.profile_result.list[0][0],
                              profile.ProfilerResult)

        # 3rd run of profiling, opening a new list, using the default algorithm
        result = profile.parameter_profile(
            problem=self.problem,
            result=result,
            optimizer=self.optimizer,
            next_guess_method='fixed_step',
            profile_index=np.array([0]),
            profile_options=options)
        # check result
        self.assertIsInstance(result.profile_result.list[1][0],
                              profile.ProfilerResult)
        self.assertIsNone(result.profile_result.list[1][1])

    def test_extending_profiles(self):
        # run profiling
        result = profile.parameter_profile(
            problem=self.problem,
            result=self.result,
            optimizer=self.optimizer,
            next_guess_method='fixed_step')

        # set new bounds (knowing that one parameter stopped at the bounds
        self.problem.lb_full = -4 * np.ones(2)
        self.problem.ub_full = 4 * np.ones(2)

        # re-run profiling using new bounds
        result = profile.parameter_profile(problem=self.problem,
                                           result=result,
                                           optimizer=self.optimizer,
                                           next_guess_method='fixed_step',
                                           profile_index=np.array([1]),
                                           profile_list=0)
        # check result
        self.assertTrue(
            isinstance(result.profile_result.list[0][0],
                       profile.ProfilerResult))
        self.assertTrue(
            isinstance(result.profile_result.list[0][1],
                       profile.ProfilerResult))

    def test_approximate_profiles(self):
        """Test for the approximate profile function."""
        n_steps = 50
        assert self.result.optimize_result.list[0].hess is None
        result = profile.approximate_parameter_profile(
            problem=self.problem, result=self.result, profile_index=[1],
            n_steps=n_steps)
        profile_list = result.profile_result.list[-1]
        assert profile_list[0] is None
        assert isinstance(profile_list[1], profile.ProfilerResult)
        assert np.isclose(profile_list[1].ratio_path.max(), 1)
        assert len(profile_list[1].ratio_path) == n_steps
        assert profile_list[1].x_path.shape == (2, n_steps)

        # with pre-defined hessian
        result = deepcopy(self.result)
        result.optimize_result.list[0].hess = np.array([[2, 0], [0, 1]])
        profile.approximate_parameter_profile(
            problem=self.problem, result=result, profile_index=[1],
            n_steps=n_steps)


# dont make this a class method such that we dont optimize twice
def test_profile_with_history():
    objective = test_objective.rosen_for_sensi(max_sensi_order=2,
                                               integrated=False)['obj']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (problem, result, optimizer) = \
            create_optimization_results(objective, dim_full=5)

    profile_options = profile.ProfileOptions(min_step_size=0.0005,
                                             delta_ratio_max=0.05,
                                             default_step_size=0.005,
                                             ratio_min=0.03)

    problem.fix_parameters([0, 3], [result.optimize_result.list[0].x[0],
                                    result.optimize_result.list[0].x[3]])
    problem.objective.history = pypesto.MemoryHistory({'trace_record': True})
    profile.parameter_profile(
        problem=problem,
        result=result,
        optimizer=optimizer,
        profile_index=np.array([0, 2, 4]),
        result_index=0,
        profile_options=profile_options
    )


@close_fig
def test_profile_with_fixed_parameters():
    """Test using profiles with fixed parameters."""
    obj = test_objective.rosen_for_sensi(max_sensi_order=1)['obj']

    lb = -2 * np.ones(5)
    ub = 2 * np.ones(5)
    problem = pypesto.Problem(
        objective=obj, lb=lb, ub=ub,
        x_fixed_vals=[0.5, -1.8], x_fixed_indices=[0, 3])

    optimizer = optimize.ScipyOptimizer(options={'maxiter': 50})
    result = optimize.minimize(
        problem=problem, optimizer=optimizer, n_starts=2)

    for i_method, next_guess_method in enumerate([
            'fixed_step', 'adaptive_step_order_0',
            'adaptive_step_order_1', 'adaptive_step_regression']):
        print(next_guess_method)
        profile.parameter_profile(
            problem=problem, result=result, optimizer=optimizer,
            next_guess_method=next_guess_method)

        # standard plotting
        axes = visualize.profiles(result, profile_list_ids=i_method)
        assert len(axes) == 3
        visualize.profile_cis(result, profile_list=i_method)

    # test profiling with all parameters fixed but one
    problem.fix_parameters([2, 3, 4],
                           result.optimize_result.list[0]['x'][2:5])
    profile.parameter_profile(
        problem=problem, result=result, optimizer=optimizer,
        next_guess_method='adaptive_step_regression')


def create_optimization_results(objective, dim_full=2):
    # create optimizer, pypesto problem and options
    options = {
        'maxiter': 200
    }
    optimizer = optimize.ScipyOptimizer(method='l-bfgs-b', options=options)

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
        options=optimize_options
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
        xs=xs, ratios=ratios, confidence_ratio=0.27)

    # correct interpolation
    assert np.isclose(lb, -3 + (-1 - (-3)) * 0.7)

    # exact pick
    assert np.isclose(ub, 3)

    lb, ub = profile.calculate_approximate_ci(
        xs=xs, ratios=ratios, confidence_ratio=0.15)

    # double value
    assert np.isclose(ub, 7)

    lb, ub = profile.calculate_approximate_ci(
        xs=xs, ratios=ratios, confidence_ratio=0.1)

    # bound value
    assert np.isclose(lb, -3)
    assert np.isclose(ub, 9)
