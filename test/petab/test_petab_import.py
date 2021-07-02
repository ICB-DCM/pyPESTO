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

from .petab_util import folder_base
from .test_sbml_conversion import ATOL, RTOL


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

    def test_check_gradients(self):
        # check gradients of simple model (should always be a true positive)
        model_name = "Bachmann_MSB2011"
        petab_problem = pypesto.petab.PetabImporter.from_yaml(
            os.path.join(folder_base, model_name, model_name + '.yaml'))
        self.assertTrue(petab_problem.check_gradients())


def test_plist_mapping():
    """Test that the AMICI objective created via PEtab correctly maps
    gradient entries when some parameters are not estimated (realized via
    edata.plist)."""
    model_name = "Boehm_JProteomeRes2014"
    petab_problem = pypesto.petab.PetabImporter.from_yaml(
        os.path.join(folder_base, model_name, model_name + '.yaml'))

    # define test parameter
    par = np.asarray(petab_problem.petab_problem.x_nominal_scaled)

    problem = petab_problem.create_problem()
    objective = problem.objective
    objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_forward)
    objective.amici_solver.setAbsoluteTolerance(1e-10)
    objective.amici_solver.setRelativeTolerance(1e-12)

    df = objective.check_grad_multi_eps(par[problem.x_free_indices],
                                        multi_eps=[1e-3, 1e-4, 1e-5])
    print("relative errors gradient: ", df.rel_err.values)
    print("absolute errors gradient: ", df.abs_err.values)
    assert np.all((df.rel_err.values < RTOL) |
                  (df.abs_err.values < ATOL))


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
