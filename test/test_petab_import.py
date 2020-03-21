
"""
This is for testing the petab import.
"""

import os
import unittest
import numpy as np

import petab
import pypesto
from test.petab_util import folder_base


class PetabImportTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.petab_problems = []
        cls.petab_importers = []
        cls.obj_edatas = []

    def test_0_import(self):
        for model_name in ["Zheng_PNAS2012", "Boehm_JProteomeRes2014"]:
            # test yaml import for one model:
            yaml_config = os.path.join(folder_base, model_name,
                                       model_name + '.yaml')
            petab_problem = petab.Problem.from_yaml(yaml_config)
            self.petab_problems.append(petab_problem)

    def test_1_compile(self):
        for petab_problem in self.petab_problems:
            importer = pypesto.PetabImporter(petab_problem)
            self.petab_importers.append(importer)

            # check model
            model = importer.create_model(force_compile=False)

            # observable ids
            model_obs_ids = list(model.getObservableIds())
            problem_obs_ids = list(petab_problem.get_observable_ids())
            self.assertEqual(set(model_obs_ids),
                             set(problem_obs_ids))

            # also other checks would be possible here

    def test_2_simulate(self):
        for petab_importer in self.petab_importers:
            obj = petab_importer.create_objective()
            edatas = petab_importer.create_edatas()
            self.obj_edatas.append((obj, edatas))

            # run function
            x_nominal = petab_importer.petab_problem.x_nominal_scaled
            ret = obj(x_nominal)

            self.assertTrue(np.isfinite(ret))

    def test_3_optimize(self):
        # run optimization
        for obj_edatas, importer in \
                zip(self.obj_edatas, self.petab_importers):
            obj = obj_edatas[0]
            optimizer = pypesto.ScipyOptimizer()
            problem = importer.create_problem(obj)
            result = pypesto.minimize(
                problem=problem, optimizer=optimizer, n_starts=2)

            self.assertTrue(np.isfinite(
                result.optimize_result.get_for_key('fval')[0]))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(PetabImportTest())
    unittest.main()
