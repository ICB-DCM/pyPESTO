"""
This is for testing the pypesto.Objective.
"""

from pypesto.objective.amici_objective import add_sim_grad_to_opt_grad
import unittest


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


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(AmiciObjectiveTest())
    unittest.main()
