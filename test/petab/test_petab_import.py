"""
This is for testing the petab import.
"""

import logging
import os
import unittest

import amici
import benchmark_models_petab as models
import numpy as np
import petab.v1 as petab
import petabtests
import pytest

import pypesto
import pypesto.optimize
import pypesto.petab
from pypesto.petab import PetabImporter

from .test_sbml_conversion import ATOL, RTOL

# In CI, bionetgen is installed here
BNGPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "BioNetGen-2.8.5")
)
if "BNGPATH" not in os.environ:
    logging.warning(f"Env var BNGPATH was not set. Setting to {BNGPATH}")
    os.environ["BNGPATH"] = BNGPATH


class PetabImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.petab_problems = []
        cls.petab_importers = []
        cls.obj_edatas = []

    def test_0_import(self):
        for model_name in ["Zheng_PNAS2012", "Boehm_JProteomeRes2014"]:
            # test yaml import for one model:
            yaml_config = os.path.join(
                models.MODELS_DIR, model_name, model_name + ".yaml"
            )
            petab_problem = petab.Problem.from_yaml(yaml_config)
            self.petab_problems.append(petab_problem)

    def test_1_compile(self):
        for petab_problem in self.petab_problems:
            importer = pypesto.petab.PetabImporter(petab_problem)
            self.petab_importers.append(importer)

            # check model
            model = importer.create_factory().create_model(force_compile=False)

            # observable ids
            model_obs_ids = list(model.getObservableIds())
            problem_obs_ids = list(petab_problem.get_observable_ids())
            self.assertEqual(set(model_obs_ids), set(problem_obs_ids))

            # also other checks would be possible here

    def test_2_simulate(self):
        for petab_importer in self.petab_importers:
            factory = petab_importer.create_factory()
            obj = factory.create_objective()
            edatas = factory.create_edatas()
            self.obj_edatas.append((obj, edatas))

            # run function
            x_nominal = factory.petab_problem.x_nominal_scaled
            ret = obj(x_nominal)

            self.assertTrue(np.isfinite(ret))

    def test_3_startpoints(self):
        # test startpoint sampling
        for obj_edatas, importer in zip(self.obj_edatas, self.petab_importers):
            obj = obj_edatas[0]
            problem = importer.create_problem(obj)

            # test for original problem
            original_dim = problem.dim
            startpoints = problem.startpoint_method(
                n_starts=2, problem=problem
            )
            self.assertEqual(startpoints.shape, (2, problem.dim))

            # test with fixed parameters
            problem.fix_parameters(0, 1)
            self.assertEqual(problem.dim, original_dim - 1)
            startpoints = problem.startpoint_method(
                n_starts=2, problem=problem
            )
            self.assertEqual(startpoints.shape, (2, problem.dim))

    def test_4_optimize(self):
        # run optimization
        for obj_edatas, importer in zip(self.obj_edatas, self.petab_importers):
            obj = obj_edatas[0]
            optimizer = pypesto.optimize.ScipyOptimizer(
                options={"maxiter": 10}
            )
            problem = importer.create_problem(obj)
            problem.startpoint_method = importer.create_startpoint_method()
            result = pypesto.optimize.minimize(
                problem=problem,
                optimizer=optimizer,
                n_starts=2,
                progress_bar=False,
            )

            self.assertTrue(np.isfinite(result.optimize_result.fval[0]))

    def test_check_gradients(self):
        """Test objective FD-gradient check function."""
        # Check gradients of simple model (should always be a true positive)
        model_name = "Bachmann_MSB2011"
        importer = pypesto.petab.PetabImporter.from_yaml(
            os.path.join(models.MODELS_DIR, model_name, model_name + ".yaml")
        )

        objective = importer.create_factory().create_objective()
        objective.amici_solver.setSensitivityMethod(
            amici.SensitivityMethod_forward
        )
        objective.amici_solver.setAbsoluteTolerance(1e-10)
        objective.amici_solver.setRelativeTolerance(1e-12)

        self.assertFalse(
            objective.check_gradients_match_finite_differences(
                multi_eps=[1e-3, 1e-4, 1e-5]
            )
        )


def test_plist_mapping():
    """Test that the AMICI objective created via PEtab correctly maps
    gradient entries when some parameters are not estimated (realized via
    edata.plist)."""
    model_name = "Boehm_JProteomeRes2014"
    petab_problem = pypesto.petab.PetabImporter.from_yaml(
        os.path.join(models.MODELS_DIR, model_name, model_name + ".yaml")
    )

    # define test parameter
    par = np.asarray(petab_problem.petab_problem.x_nominal_scaled)

    problem = petab_problem.create_problem()
    objective = problem.objective
    # check that x_names are correctly subsetted
    assert objective.x_names == [
        problem.x_names[ix] for ix in problem.x_free_indices
    ]
    objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_forward
    )
    objective.amici_solver.setAbsoluteTolerance(1e-10)
    objective.amici_solver.setRelativeTolerance(1e-12)

    df = objective.check_grad_multi_eps(
        par[problem.x_free_indices], multi_eps=[1e-3, 1e-4, 1e-5]
    )
    print("relative errors gradient: ", df.rel_err.values)
    print("absolute errors gradient: ", df.abs_err.values)
    assert np.all((df.rel_err.values < RTOL) | (df.abs_err.values < ATOL))


def test_max_sensi_order():
    """Test that the AMICI objective created via PEtab exposes derivatives
    correctly."""
    model_name = "Boehm_JProteomeRes2014"
    importer = pypesto.petab.PetabImporter.from_yaml(
        os.path.join(models.MODELS_DIR, model_name, model_name + ".yaml")
    )

    # define test parameter
    par = importer.petab_problem.x_nominal_scaled
    npar = len(par)

    # auto-computed max_sensi_order and fim_for_hess
    objective = importer.create_factory().create_objective()
    hess = objective(par, sensi_orders=(2,))
    assert hess.shape == (npar, npar)
    assert (hess != 0).any()
    objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_adjoint
    )
    with pytest.raises(ValueError):
        objective(par, sensi_orders=(2,))
    objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_forward
    )

    # fix max_sensi_order to 1
    objective = importer.create_factory().create_objective(max_sensi_order=1)
    objective(par, sensi_orders=(1,))
    with pytest.raises(ValueError):
        objective(par, sensi_orders=(2,))

    # do not use FIM
    objective = importer.create_factory().create_objective(fim_for_hess=False)
    with pytest.raises(ValueError):
        objective(par, sensi_orders=(2,))

    # only allow computing function values
    objective = importer.create_factory().create_objective(max_sensi_order=0)
    objective(par)
    with pytest.raises(ValueError):
        objective(par, sensi_orders=(1,))


def test_petab_pysb_optimization():
    test_case = "0001"
    test_case_dir = petabtests.get_case_dir(
        test_case, version="v2.0.0", format_="pysb"
    )
    petab_yaml = test_case_dir / petabtests.problem_yaml_name(test_case)
    # expected results
    solution = petabtests.load_solution(
        test_case, format="pysb", version="v2.0.0"
    )

    petab_problem = petab.Problem.from_yaml(petab_yaml)
    importer = PetabImporter(petab_problem)
    problem = importer.create_problem()

    # ensure simulation result for true parameters matches
    assert np.isclose(
        problem.objective(petab_problem.x_nominal), -solution[petabtests.LLH]
    )

    optimizer = pypesto.optimize.ScipyOptimizer()
    result = pypesto.optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=10,
        progress_bar=False,
    )
    fvals = np.array(result.optimize_result.fval)

    # ensure objective after optimization is not worse than for true parameters
    assert np.all(fvals <= -solution[petabtests.LLH])


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(PetabImportTest())
    unittest.main()
