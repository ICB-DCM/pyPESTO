"""
This is for testing profile based validation intervals.
"""

import unittest

import numpy as np

import pypesto
import pypesto.optimize as optimize
import pypesto.profile as profile


class ValidationIntervalTest(unittest.TestCase):

    @classmethod
    def setUp(cls):

        cls.lb = np.array([-1])
        cls.ub = np.array([5])

        cls.problem_training_data = pypesto.Problem(
            lsq_residual_objective(0),
            cls.lb, cls.ub)

        cls.problem_all_data = pypesto.Problem(
            pypesto.objective.AggregatedObjective(
                [lsq_residual_objective(0),
                 lsq_residual_objective(2)]),
            cls.lb, cls.ub)

        # optimum f(0)=0
        cls.result_training_data = optimize.minimize(cls.problem_training_data,
                                                     n_starts=5, filename=None)
        # Optimum f(1)=2
        cls.result_all_data = optimize.minimize(cls.problem_all_data,
                                                n_starts=5, filename=None)

    def test_validation_intervals(self):
        """Test validation profiles."""

        # fit with handing over all data
        profile.validation_profile_significance(self.problem_all_data,
                                                self.result_training_data,
                                                self.result_all_data)

        # fit with refitting inside function
        res_with_significance_true = profile.validation_profile_significance(
            self.problem_all_data,
            self.result_training_data)

        # test return_significance = False leads the result to 1-significance
        res_with_significance_false = profile.validation_profile_significance(
            self.problem_all_data,
            self.result_training_data,
            return_significance=False)

        self.assertAlmostEqual(1-res_with_significance_true,
                               res_with_significance_false,
                               places=4)

        # test if significance=1, if the validation experiment coincides with
        # the Max Likelihood prediction

        problem_val = pypesto.Problem(pypesto.objective.AggregatedObjective(
                [lsq_residual_objective(0),
                 lsq_residual_objective(0)]),
            self.lb, self.ub)

        significance = profile.validation_profile_significance(
            problem_val,
            self.result_training_data)

        self.assertAlmostEqual(significance, 1)

        # Test if the significance approaches zero (monotonously decreasing)
        # if the discrepancy between Max. Likelihood prediction and observed
        # validation interval is increased

        for d_val in [0.1, 1, 10, 100]:

            # problem including d_val
            problem_val = pypesto.Problem(
                pypesto.objective.AggregatedObjective(
                    [lsq_residual_objective(0),
                     lsq_residual_objective(d_val)]),
                self.lb, self.ub)

            sig_d_val = profile.validation_profile_significance(
                problem_val,
                self.result_training_data)

            # test, if the more extreme measurement leads to a more
            # significant result
            self.assertLessEqual(sig_d_val, significance)
            significance = sig_d_val

        # test if the significance approaches 0 for d_val=100
        self.assertAlmostEqual(significance, 0)


def lsq_residual_objective(d: float):
    """
    Returns an objective for the function

    f(x) = (x-d)^2
    """
    def f(x):
        return np.sum((x[0]-d)**2)

    def grad(x):
        return 2 * (x-d)

    return pypesto.Objective(fun=f, grad=grad)
