import numpy as np
import unittest

import pypesto
import test.test_objective as test_objective
from test.util import folder_base, model_names


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
        pypesto.minimize(problem=problem, n_starts=9, engine=engine)

    def test_petab(self):
        for engine in [pypesto.MultiProcessEngine()]:
            self._test_petab(engine)

    def _test_petab(self, engine):
        petab_importer = pypesto.PetabImporter.from_folder(
            folder_base + model_names[0])
        objective = petab_importer.create_objective()
        problem = petab_importer.create_problem(objective)
        pypesto.minimize(problem=problem, n_starts=3, engine=engine)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(EngineTest())
    unittest.main()
