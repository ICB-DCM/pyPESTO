
"""
This is for testing the petab import.
"""

import unittest
import git
import os

import petab
import pypesto


# prerequisites

# clone or pull git repo

repo_base = "doc/example/tmp/benchmark-models/"
try:
    git.Git().clone("git://github.com/LoosC/Benchmark-Models.git",
                    repo_base, depth=1)
except Exception:
    git.Git(repo_base).pull()

# model folder base
folder_base = repo_base + "hackathon_contributions_new_data_format/"

# model name
model_names = ["Zheng_PNAS2012", "Boehm_JProteomeRes2014"]


class PetabImportTest(unittest.TestCase):
    
    def setUp(self):
        self.petab_problems = []
        for model_name in model_names:
            petab_problem = petab.Problem.from_folder(
                folder_base + model_name)
            self.petab_problems.append(petab_problem)
        
        self.petab_importers = []
        self.obj_edatas = []

    def test_0_compile(self):
        for petab_problem in self.petab_problems:
            importer = pypesto.PetabImporter(petab_problem,
                                             force_compile=True)
            self.petab_importers.append(importer)

            # check model
            model = importer.model

            # observable ids
            model_obs_ids = list(model.getObservableIds())
            problem_obs_ids = list(petab_problem.get_observables().keys())
            self.assertEqual(set(model_obs_ids),
                             set(problem_obs_ids))

            # TODO continue

    def test_1_simulate(self):
        for petab_importer in self.petab_importers:
            obj, edatas = petab_importer.create_objective()
            self.obj_edatas.append((obj, edatas))
            
            # run function
            x_nominal = petab_importer.petab_problem.x_nominal
            ret = obj(x_nominal)
            
            self.assertTrue(np.isfinite(ret))

    def test_2_optimize(self):
        # run optimization
        for obj, edatas in self.obj_edatas:
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
