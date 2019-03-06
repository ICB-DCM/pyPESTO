"""
This is for testing the pypesto.Objective.
"""

from pypesto.objective.amici_objective import add_sim_grad_to_opt_grad
import unittest

import petab
import pypesto
import pypesto.objective.constants
import numpy as np
from test.petab_util import folder_base

ATOL = 1e-6
RTOL = 1e-6


class AmiciObjectiveTest(unittest.TestCase):

    def test_add_sim_grad_to_opt_grad(self):
        """
        Test gradient mapping/summation works as expected.
        """
        par_opt_ids = ['opt_par_1',
                       'opt_par_2',
                       'opt_par_3']
        mapping_par_opt_to_par_sim = \
            [
                'opt_par_1',
                'opt_par_3',
                'opt_par_3'
            ]

        sim_grad = [1.0, 3.0, 5.0]
        opt_grad = [1.0, 1.0, 1.0]
        expected = [3.0, 1.0, 17.0]

        add_sim_grad_to_opt_grad(
            par_opt_ids,
            mapping_par_opt_to_par_sim,
            sim_grad,
            opt_grad,
            coefficient=2.0)

        self.assertEqual(expected, opt_grad)

    def test_preeq_guesses(self):
        """
        Test whether optimization with preequilibration guesses works, asserts
        that steadystate guesses are written and checks that gradient is still
        correct with guesses set
        """
        petab_problem = petab.Problem.from_folder(folder_base +
                                                  "Zheng_PNAS2012")
        petab_problem.model_name = "Zheng_PNAS2012"
        importer = pypesto.PetabImporter(petab_problem)
        obj = importer.create_objective()
        problem = importer.create_problem(obj)
        optimizer = pypesto.ScipyOptimizer('ls_trf')

        result = pypesto.minimize(
            problem=problem, optimizer=optimizer, n_starts=2,
        )

        self.assertTrue(obj.steadystate_guesses['fval'] < np.inf)
        self.assertTrue(len(obj.steadystate_guesses['data']) == 1)

        df = obj.check_grad(
            result.optimize_result.list[0]['x'],
            eps=1e-3,
            verbosity=0,
            mode=pypesto.objective.constants.MODE_FUN
        )
        print("relative errors MODE_FUN: ", df.rel_err.values)
        print("absolute errors MODE_FUN: ", df.abs_err.values)
        self.assertTrue(np.all((df.rel_err.values < RTOL) |
                               (df.abs_err.values < ATOL)))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(AmiciObjectiveTest())
    unittest.main()
