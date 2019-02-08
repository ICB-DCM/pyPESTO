import numpy as np
import unittest

import pypesto
import test.test_objective as test_objective


class EngineTest(unittest.TestCase):

    def test_basic(self):
        for engine in [pypesto.SingleCoreEngine(),
                       pypesto.MultiProcessEngine(),
                       pypesto.MultiProcessEngine(5)]:
            self._test_basic(engine)

    def _test_basic(self, engine):
        # set up problem
        objective = test_objective.rosen_for_sensi(max_sensi_order=2)['obj']
        lb = 0 * np.ones((1, 2))
        ub = 1 * np.ones((1, 2))
        problem = pypesto.Problem(objective, lb, ub)
        result = pypesto.minimize(problem=problem, n_starts=9, engine=engine)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(EngineTest())
    unittest.main()
