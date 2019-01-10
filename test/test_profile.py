"""
This is for testing profiling of the pypesto.Objective.
"""

import numpy as np
import pypesto
import unittest
import test.test_objective as test_objective
import warnings
import copy


class ProfilerTest(unittest.TestCase):

    def runTest(self):
        objective = test_objective.rosen_for_sensi(max_sensi_order=2,
                                                   integrated=True)['obj']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            (problem, result, optimizer) = \
                create_optimization_results(objective)

            result1 = copy.deepcopy(result)
            result2 = copy.deepcopy(result)

            # profiling works using default settings and different proposals
            # for creating the next starting point
            self.check_default_profiling(problem, result1, optimizer)

            # profiling works when specifying additional options
            self.check_selected_profiling(problem, result2, optimizer)

            # extending profiles (when changing bounds) works (T.B.D.)

    def check_default_profiling(self, problem, result, optimizer):
        # loop over  methods for creating new initial guesses
        method_list = [None, 'fixed_step', 'adaptive_step_order_0']

        for method in method_list:
            # run profiling
            result = pypesto.profile(problem=problem,
                                     result=result,
                                     optimizer=optimizer,
                                     next_guess_method=method)

            # check result
            self.assertTrue(
                isinstance(result.profile_result.list[0][0],
                           pypesto.ProfilerResult))

    def check_selected_profiling(self, problem, result, optimizer):
        # 1st run of profiling, computing just one out of two profiles
        result = pypesto.profile(problem=problem,
                                 result=result,
                                 optimizer=optimizer,
                                 profile_index=np.array([0, 1]),
                                 next_guess_method='fixed_step',
                                 result_index=1)

        self.assertIsInstance(result.profile_result.list[0][1],
                              pypesto.ProfilerResult)
        self.assertIsNone(result.profile_result.list[0][0])

        # 2nd run of profiling, appending to an existing list of profiles
        # using another algorithm
        result = pypesto.profile(problem=problem,
                                 result=result,
                                 optimizer=optimizer,
                                 profile_index=np.array([1, 0]),
                                 next_guess_method='adaptive_step_order_0',
                                 result_index=2,
                                 profile_list=0)

        self.assertIsInstance(result.profile_result.list[0][0],
                              pypesto.ProfilerResult)

        # 3rd run of profiling, opening a new list, using the default algorithm
        result = pypesto.profile(problem=problem,
                                 result=result,
                                 optimizer=optimizer,
                                 profile_index=np.array([1, 0]))
        # check result
        self.assertIsInstance(result.profile_result.list[1][0],
                              pypesto.ProfilerResult)
        self.assertIsNone(result.profile_result.list[1][1])


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
