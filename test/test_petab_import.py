"""
This is for testing the petab import.
"""

import os
import unittest
import numpy as np
import pytest

import amici
import petab
import pypesto
import pypesto.optimize
import pypesto.petab
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
            importer = pypesto.petab.PetabImporter(petab_problem)
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
            optimizer = pypesto.optimize.ScipyOptimizer(
                options={'maxiter': 10})
            problem = importer.create_problem(obj)
            result = pypesto.optimize.minimize(
                problem=problem, optimizer=optimizer, n_starts=2)

            self.assertTrue(np.isfinite(
                result.optimize_result.get_for_key('fval')[0]))


def test_max_sensi_order():
    """Test that the AMICI objective created via PEtab exposes derivatives
    correctly."""
    model_name = "Boehm_JProteomeRes2014"
    problem = pypesto.petab.PetabImporter.from_yaml(
        os.path.join(folder_base, model_name, model_name + '.yaml'))

    # define test parameter
    par = problem.petab_problem.x_nominal_scaled
    npar = len(par)

    # auto-computed max_sensi_order and fim_for_hess
    objective = problem.create_objective()
    hess = objective(par, sensi_orders=(2,))
    assert hess.shape == (npar, npar)
    assert (hess != 0).any()
    objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_adjoint)
    with pytest.raises(ValueError):
        objective(par, sensi_orders=(2,))
    objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_forward)

    # fix max_sensi_order to 1
    objective = problem.create_objective(max_sensi_order=1)
    objective(par, sensi_orders=(1,))
    with pytest.raises(ValueError):
        objective(par, sensi_orders=(2,))

    # do not use FIM
    objective = problem.create_objective(fim_for_hess=False)
    with pytest.raises(ValueError):
        objective(par, sensi_orders=(2,))

    # only allow computing function values
    objective = problem.create_objective(max_sensi_order=0)
    objective(par)
    with pytest.raises(ValueError):
        objective(par, sensi_orders=(1,))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(PetabImportTest())
    unittest.main()
