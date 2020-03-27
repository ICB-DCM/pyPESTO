import numpy as np
import unittest
import copy
import cloudpickle as pickle

import pypesto
import test.test_objective as test_objective
from test.petab_util import folder_base
import amici


class EngineTest(unittest.TestCase):

    def test_basic(self):
        for engine in [pypesto.SingleCoreEngine(),
                       pypesto.MultiProcessEngine(n_procs=2),
                       pypesto.MultiThreadEngine(n_procs=8)]:
            self._test_basic(engine)

    def _test_basic(self, engine):
        # set up problem
        objective = test_objective.rosen_for_sensi(max_sensi_order=2)['obj']
        lb = 0 * np.ones((1, 2))
        ub = 1 * np.ones((1, 2))
        problem = pypesto.Problem(objective, lb, ub)
        optimizer = pypesto.ScipyOptimizer(options={'maxiter': 10})
        result = pypesto.minimize(
            problem=problem, n_starts=5, engine=engine, optimizer=optimizer)
        self.assertTrue(len(result.optimize_result.as_list()) == 5)

    def test_petab(self):
        for engine in [pypesto.SingleCoreEngine(),
                       pypesto.MultiProcessEngine(n_procs=2),
                       pypesto.MultiThreadEngine(n_procs=8)]:
            self._test_petab(engine)

    def _test_petab(self, engine):
        petab_importer = pypesto.PetabImporter.from_yaml(
            folder_base + "Zheng_PNAS2012/Zheng_PNAS2012.yaml")
        objective = petab_importer.create_objective()
        problem = petab_importer.create_problem(objective)
        optimizer = pypesto.ScipyOptimizer(options={'maxiter': 5})
        result = pypesto.minimize(
            problem=problem, n_starts=5, engine=engine, optimizer=optimizer)
        self.assertTrue(len(result.optimize_result.as_list()) == 5)

    @staticmethod
    def test_deepcopy_objective():
        """Test copying objectives (needed for MultiProcessEngine)."""
        petab_importer = pypesto.PetabImporter.from_yaml(
            folder_base + "Zheng_PNAS2012/Zheng_PNAS2012.yaml")
        objective = petab_importer.create_objective()

        objective.amici_solver.setSensitivityMethod(
            amici.SensitivityMethod_adjoint)

        objective2 = copy.deepcopy(objective)

        # test some properties
        assert objective.amici_model.getParameterIds() \
            == objective2.amici_model.getParameterIds()
        assert objective.amici_solver.getSensitivityOrder() \
            == objective2.amici_solver.getSensitivityOrder()
        assert objective.amici_solver.getSensitivityMethod() \
            == objective2.amici_solver.getSensitivityMethod()
        assert len(objective.edatas) == len(objective2.edatas)

        assert objective.amici_model is not objective2.amici_model
        assert objective.amici_solver is not objective2.amici_solver
        assert objective.steadystate_guesses is not objective2.steadystate_guesses

    @staticmethod
    def test_pickle_objective():
        """Test serializing objectives (needed for MultiThreadEngine)."""
        petab_importer = pypesto.PetabImporter.from_yaml(
            folder_base + "Zheng_PNAS2012/Zheng_PNAS2012.yaml")
        objective = petab_importer.create_objective()

        objective.amici_solver.setSensitivityMethod(
            amici.SensitivityMethod_adjoint)

        objective2=pickle.loads(pickle.dumps(objective))

        # test some properties
        assert objective.amici_model.getParameterIds() \
               == objective2.amici_model.getParameterIds()
        assert objective.amici_solver.getSensitivityOrder() \
               == objective2.amici_solver.getSensitivityOrder()
        # TODO Pickling does not preserve attributes yet
        #assert objective.amici_solver.getSensitivityMethod() \
        #       == objective2.amici_solver.getSensitivityMethod()
        assert len(objective.edatas) == len(objective2.edatas)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(EngineTest())
    unittest.main()
