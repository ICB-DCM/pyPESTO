"""
This is for testing profiling of the pypesto.Objective.
"""

import numpy as np
import pypesto
import unittest
import test.test_objective as test_objective
import warnings

from pypesto.objective import Objective


class ProfilerTest(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.objective: Objective = test_objective.rosen_for_sensi(
            max_sensi_order=2, integrated=True
        )['obj']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (cls.problem, cls.result, cls.optimizer) = \
                create_optimization_results(cls.objective)

    def test_default_profiling(self):
        # loop over  methods for creating new initial guesses
        method_list = [None, 'fixed_step', 'adaptive_step_order_0',
                       'adaptive_step_order_1', 'adaptive_step_regression']
        for method in method_list:
            # run profiling
            result = pypesto.parameter_profile(problem=self.problem,
                                               result=self.result,
                                               optimizer=self.optimizer,
                                               next_guess_method=method)

            # check result
            self.assertTrue(
                isinstance(result.profile_result.list[0][0],
                           pypesto.ProfilerResult))

            # check whether profiling needed maybe too many steps
            steps = result.profile_result.list[0][0]['ratio_path'].size
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

    def test_selected_profiling(self):
        # create options in order to ensure a short computation time
        options = pypesto.ProfileOptions(default_step_size=0.02,
                                         min_step_size=0.005,
                                         max_step_size=1.,
                                         step_size_factor=1.5,
                                         delta_ratio_max=0.2,
                                         ratio_min=0.3,
                                         reg_points=5,
                                         reg_order=2)

        # 1st run of profiling, computing just one out of two profiles
        result = pypesto.parameter_profile(problem=self.problem,
                                           result=self.result,
                                           optimizer=self.optimizer,
                                           profile_index=np.array([0, 1]),
                                           next_guess_method='fixed_step',
                                           result_index=1,
                                           profile_options=options)

        self.assertIsInstance(result.profile_result.list[0][1],
                              pypesto.ProfilerResult)
        self.assertIsNone(result.profile_result.list[0][0])

        # 2nd run of profiling, appending to an existing list of profiles
        # using another algorithm and another optimum
        result = pypesto.parameter_profile(problem=self.problem,
                                           result=result,
                                           optimizer=self.optimizer,
                                           profile_index=np.array([1, 0]),
                                           result_index=2,
                                           profile_list=0,
                                           profile_options=options)

        self.assertIsInstance(result.profile_result.list[0][0],
                              pypesto.ProfilerResult)

        # 3rd run of profiling, opening a new list, using the default algorithm
        result = pypesto.parameter_profile(problem=self.problem,
                                           result=result,
                                           optimizer=self.optimizer,
                                           next_guess_method='fixed_step',
                                           profile_index=np.array([1, 0]),
                                           profile_options=options)
        # check result
        self.assertIsInstance(result.profile_result.list[1][0],
                              pypesto.ProfilerResult)
        self.assertIsNone(result.profile_result.list[1][1])

    def test_extending_profiles(self):
        # run profiling
        result = pypesto.parameter_profile(problem=self.problem,
                                           result=self.result,
                                           optimizer=self.optimizer,
                                           next_guess_method='fixed_step')

        # set new bounds (knowing that one parameter stopped at the bounds
        self.problem.lb = -4 * np.ones((1, 2))
        self.problem.ub = 4 * np.ones((1, 2))

        # re-run profiling using new bounds
        result = pypesto.parameter_profile(problem=self.problem,
                                           result=result,
                                           optimizer=self.optimizer,
                                           next_guess_method='fixed_step',
                                           profile_index=np.array([0, 1]),
                                           profile_list=0)
        # check result
        self.assertTrue(
            isinstance(result.profile_result.list[0][0],
                       pypesto.ProfilerResult))
        self.assertTrue(
            isinstance(result.profile_result.list[0][1],
                       pypesto.ProfilerResult))


# dont make this a class method such that we dont optimize twice
def test_profile_with_history():
    objective = test_objective.rosen_for_sensi(max_sensi_order=2,
                                               integrated=False)['obj']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (problem, result, optimizer) = \
            create_optimization_results(objective)

    profile_options = pypesto.ProfileOptions(min_step_size=0.0005,
                                             delta_ratio_max=0.05,
                                             default_step_size=0.005,
                                             ratio_min=0.03)

    problem.objective.history = pypesto.MemoryHistory({'trace_record': True})
    pypesto.parameter_profile(
        problem=problem,
        result=result,
        optimizer=optimizer,
        profile_index=np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]),
        result_index=0,
        profile_options=profile_options
    )


def create_optimization_results(objective):
    # create optimizer, pypesto problem and options
    options = {
        'maxiter': 200
    }
    optimizer = pypesto.ScipyOptimizer(method='TNC',
                                       options=options)

    lb = -2 * np.ones((1, 2))
    ub = 2 * np.ones((1, 2))
    problem = pypesto.Problem(objective, lb, ub)

    optimize_options = pypesto.OptimizeOptions(allow_failed_starts=True)

    # run optimization
    result = pypesto.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=5,
        startpoint_method=pypesto.startpoint.uniform,
        options=optimize_options
    )

    return problem, result, optimizer
