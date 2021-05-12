"""
This is for testing profile based validation intervals.
"""

import numpy as np
import unittest


import pypesto
import pypesto.optimize as optimize
import pypesto.profile as profile


class ValidationIntervalTest(unittest.TestCase):

    @classmethod
    def setUp(cls):

        lb = np.array([-1])
        ub = np.array([5])

        cls.problem_training_data = pypesto.Problem(
            lsq_residual_objective(0),
            lb, ub)

        cls.problem_all_data = pypesto.Problem(
            pypesto.objective.AggregatedObjective(
                [lsq_residual_objective(0),
                 lsq_residual_objective(2)]),
            lb, ub)

        # optimum f(0)=0
        cls.result_training_data = optimize.minimize(cls.problem_training_data,
                                                     n_starts=5)
        # Optimum f(1)=2
        cls.result_all_data = optimize.minimize(cls.problem_all_data,
                                                n_starts=5)

    def test_validation_intervals(self):
        """Test validation profiles."""

        # fit with handing over all data
        profile.validation_profile_significance(self.problem_all_data,
                                                self.result_training_data,
                                                self.result_all_data)

        # fit with refitting inside function
        profile.validation_profile_significance(self.problem_all_data,
                                                self.result_training_data)


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
