
"""
This is for testing the petab import.
"""

import unittest
import numpy as np

import petab
import pypesto
from test.petab_util import folder_base
import amici


class PetabImportTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.petab_problems = []
        cls.petab_importers = []
        cls.obj_edatas = []

    def test_0_import(self):
        for model_name in ["Zheng_PNAS2012", "Boehm_JProteomeRes2014"]:
            petab_problem = petab.Problem.from_folder(
                folder_base + model_name)
            self.petab_problems.append(petab_problem)

    def test_1_compile(self):
        for petab_problem in self.petab_problems:
            importer = pypesto.PetabImporter(petab_problem)
            self.petab_importers.append(importer)

            # check model
            model = importer.create_model(force_compile=True)

            # observable ids
            model_obs_ids = list(model.getObservableIds())
            problem_obs_ids = list(petab_problem.get_observables().keys())
            self.assertEqual(set(model_obs_ids),
                             set(problem_obs_ids))

            # also other checks would be possible here

    def test_2_simulate(self):
        for petab_importer in self.petab_importers:
            obj = petab_importer.create_objective()
            edatas = petab_importer.create_edatas()
            self.obj_edatas.append((obj, edatas))

            # run function
            x_nominal = petab_importer.petab_problem.x_nominal
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


class SpecialFeaturesTest(unittest.TestCase):

    def test_replicates(self):
        """
        Use a model that has replicates and check that all data points are
        inserted.
        """
        # import a model with replicates at some time points and observables
        importer = pypesto.PetabImporter.from_folder(
            folder_base + "Schwen_PONE2014")

        # create amici.ExpData list
        edatas = importer.create_edatas()

        # convert to dataframes
        amici_df = amici.getDataObservablesAsDataFrame(
            importer.create_model(), edatas)
        # reduce to data columns
        amici_df = amici_df[[col for col in amici_df.columns
                             if col.startswith("observable_")
                             and not col.endswith("_std")]]

        meas_df = importer.petab_problem.measurement_df

        # amici_df subset measurement_df
        for _, row in meas_df.iterrows():
            val = row.measurement
            # test if amici_df contains this value somewhere
            # this will test up to np.isclose closeness
            self.assertTrue(np.isclose(amici_df, val).any(axis=1).any())

        # and the other way round
        meas_vals = meas_df.measurement
        for _, row in amici_df.iterrows():
            for val in row:
                if np.isfinite(val):
                    self.assertTrue(np.isclose(meas_vals, val).any())


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(PetabImportTest())
    suite.addTest(SpecialFeaturesTest())
    unittest.main()
