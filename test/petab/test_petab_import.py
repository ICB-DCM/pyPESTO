"""
This is for testing the petab import.
"""

import logging
import os
import unittest
from itertools import chain

import amici
import benchmark_models_petab as models
import numpy as np
import petabtests
import pytest
from petab import v2

import pypesto
import pypesto.optimize
import pypesto.petab
from pypesto.petab import PetabImporter

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
        for model_name in [
            "Zheng_PNAS2012",
            "Boehm_JProteomeRes2014",
            "Weber_BMC2015",
        ]:
            # test yaml import for one model:
            petab_problem = models.get_problem(model_name)
            self.petab_problems.append(petab_problem)

    def test_1_compile(self):
        for petab_problem in self.petab_problems:
            importer = pypesto.petab.PetabImporter(petab_problem)
            self.petab_importers.append(importer)

            # check model
            model = importer.create_objective_creator().create_model(
                force_compile=False
            )

            # observable ids
            model_obs_ids = list(model.get_observable_ids())
            problem_obs_ids = list(petab_problem.get_observable_ids())
            self.assertEqual(set(model_obs_ids), set(problem_obs_ids))

            # also other checks would be possible here

    def test_2_simulate(self):
        for petab_importer in self.petab_importers:
            factory = petab_importer.create_objective_creator()
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
        model_name = "Boehm_JProteomeRes2014"
        importer = pypesto.petab.PetabImporter.from_yaml(
            models.get_problem_yaml_path(model_name)
        )

        objective = importer.create_problem().objective
        objective.amici_solver.set_sensitivity_method(
            amici.SensitivityMethod.forward
        )
        objective.amici_solver.set_absolute_tolerance(1e-10)
        objective.amici_solver.set_relative_tolerance(1e-12)

        self.assertFalse(
            objective.check_gradients_match_finite_differences(
                multi_eps=[1e-3, 1e-4, 1e-5]
            )
        )


def test_plist_mapping():
    """Test that the AMICI objective created via PEtab correctly uses
    ExpData.plist.

    That means, ensure that
    1) it only computes gradient entries that are really required, and
    2) correctly maps model-gradient entries to the objective gradient
    3) with or without pypesto-fixed parameters.

    In Bruno_JExpBot2016, different parameters are estimated in different
    conditions.
    """
    model_name = "Bruno_JExpBot2016"
    petab_importer = pypesto.petab.PetabImporter.from_yaml(
        os.path.join(models.MODELS_DIR, model_name, model_name + ".yaml")
    )
    objective_creator = petab_importer.create_objective_creator()
    problem = petab_importer.create_problem(
        objective_creator.create_objective()
    )
    objective = problem.objective
    objective.amici_solver.set_sensitivity_method(
        amici.SensitivityMethod.forward
    )
    objective.amici_solver.set_absolute_tolerance(1e-16)
    objective.amici_solver.set_relative_tolerance(1e-15)

    # slightly perturb the parameters to avoid vanishing gradients
    par = np.asarray(petab_importer.petab_problem.x_nominal_free_scaled) * 1.01

    # call once to make sure ExpDatas are populated
    fx1, grad1 = objective(par, sensi_orders=(0, 1))

    plists1 = {edata.plist for edata in objective.edatas}
    assert len(plists1) == 6, plists1

    df = objective.check_grad_multi_eps(
        par[problem.x_free_indices], multi_eps=[1e-5, 1e-6, 1e-7, 1e-8]
    )
    print("relative errors gradient: ", df.rel_err.values)
    print("absolute errors gradient: ", df.abs_err.values)
    assert np.all((df.rel_err.values < 1e-8) | (df.abs_err.values < 1e-10))

    # do the same after fixing some parameters
    # we fix them to the previous values, so we can compare the results
    fixed_ids = ["init_b10_1", "init_bcry_1"]
    # the corresponding amici parameters (they are only ever mapped to the
    #  respective parameters in fixed_ids, and thus, should not occur in any
    #  `plist` later on)
    fixed_model_par_ids = ["init_b10", "init_bcry"]
    fixed_model_par_idxs = [
        objective.amici_model.get_parameter_ids().index(id)
        for id in fixed_model_par_ids
    ]
    fixed_idxs = [problem.x_names.index(id) for id in fixed_ids]
    problem.fix_parameters(fixed_idxs, par[fixed_idxs])
    assert objective is problem.objective
    assert problem.x_fixed_indices == fixed_idxs
    fx2, grad2 = objective(par[problem.x_free_indices], sensi_orders=(0, 1))
    assert np.isclose(fx1, fx2, rtol=1e-10, atol=1e-14)
    assert np.allclose(
        grad1[problem.x_free_indices], grad2, rtol=1e-10, atol=1e-14
    )
    plists2 = {edata.plist for edata in objective.edatas}
    # the fixed parameters should have been in plist1, but not in plist2
    assert (
        set(fixed_model_par_idxs) - set(chain.from_iterable(plists1)) == set()
    )
    assert (
        set(fixed_model_par_idxs) & set(chain.from_iterable(plists2)) == set()
    )


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
    objective = importer.create_objective_creator().create_objective()
    hess = objective(par, sensi_orders=(2,))
    assert hess.shape == (npar, npar)
    assert (hess != 0).any()
    objective.amici_solver.set_sensitivity_method(
        amici.SensitivityMethod.adjoint
    )
    with pytest.raises(ValueError):
        objective(par, sensi_orders=(2,))
    objective.amici_solver.set_sensitivity_method(
        amici.SensitivityMethod.forward
    )

    # fix max_sensi_order to 1
    objective = importer.create_objective_creator().create_objective(
        max_sensi_order=1
    )
    objective(par, sensi_orders=(1,))
    with pytest.raises(ValueError):
        objective(par, sensi_orders=(2,))

    # do not use FIM
    objective = importer.create_objective_creator().create_objective(
        fim_for_hess=False
    )
    with pytest.raises(ValueError):
        objective(par, sensi_orders=(2,))

    # only allow computing function values
    objective = importer.create_objective_creator().create_objective(
        max_sensi_order=0
    )
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

    petab_problem = v2.Problem.from_yaml(petab_yaml)
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


def test_petab_v2_boehm():
    import copy
    import pickle

    from pypesto.optimize.optimizer import ScipyOptimizer

    # load test problem
    problem_id = "Boehm_JProteomeRes2014"
    petab_problem = v2.Problem.from_yaml(
        models.get_problem_yaml_path(problem_id)
    )
    expected_fval_nominal = 138.22199693517703

    # create model
    importer = PetabImporter(petab_problem)
    problem = importer.create_problem()
    assert problem.x_names == petab_problem.x_ids
    assert problem.dim == petab_problem.n_estimated == 9
    assert problem.dim_full == len(petab_problem.parameters) == 11
    # petab-non-estimated parameters are fixed in the pypesto problem
    assert problem.x_fixed_indices == [
        i for i, p in enumerate(petab_problem.parameters) if not p.estimate
    ]
    assert isinstance(
        problem.objective, pypesto.objective.amici.amici.AmiciPetabV2Objective
    )

    # evaluate objective at nominal parameters
    fval = problem.objective(np.asarray(petab_problem.x_nominal_free))
    assert np.isclose(fval, expected_fval_nominal)

    # deepcopy works?
    problem_deepcopy = copy.deepcopy(problem)
    fval = problem_deepcopy.objective(np.asarray(petab_problem.x_nominal_free))
    assert np.isclose(fval, expected_fval_nominal)

    # pickling works?
    problem_pickled = pickle.loads(pickle.dumps(problem))  # noqa: S301
    fval = problem_pickled.objective(np.asarray(petab_problem.x_nominal_free))
    assert np.isclose(fval, expected_fval_nominal)

    # gradient works?
    fval, grad = problem.objective(
        np.asarray(petab_problem.x_nominal_free), sensi_orders=(0, 1)
    )
    assert np.isclose(fval, expected_fval_nominal)
    assert len(grad) == petab_problem.n_estimated

    # fixing parameters, ...
    problem.unfix_parameters(petab_problem.x_fixed_indices)
    assert problem.dim == len(petab_problem.parameters)
    with pytest.raises(ValueError, match="Cannot compute gradient"):
        # cannot compute sensitivities for fixed parameters
        problem.objective(
            np.asarray(petab_problem.x_nominal), sensi_orders=(0, 1)
        )
    fval = problem.objective(np.asarray(petab_problem.x_nominal))
    assert np.isclose(fval, expected_fval_nominal)
    # re-fixing parameters
    problem.fix_parameters(
        petab_problem.x_fixed_indices, petab_problem.x_nominal_fixed
    )
    assert problem.dim == petab_problem.n_estimated
    fval, grad = problem.objective(
        np.asarray(petab_problem.x_nominal_free), sensi_orders=(0, 1)
    )
    assert np.isclose(fval, expected_fval_nominal)
    assert len(grad) == petab_problem.n_estimated

    # TODO mode=res/fun,
    # TODO hess/fim...

    # single optimization works?
    optimizer = ScipyOptimizer()
    result = optimizer.minimize(
        id="1",
        problem=problem,
        x0=np.asarray(petab_problem.x_nominal_free) + 0.1,
    )
    print(result)
    assert result.fval0 is not None, result.message
    assert result.fval < result.fval0

    # multi-processing optimization works?
    result = pypesto.optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=4,
        engine=pypesto.engine.MultiProcessEngine(),
        progress_bar=False,
    )
    assert len(result.optimize_result.list) == 4
    for local_result in result.optimize_result.list:
        assert local_result.fval0 is not None, local_result.message
        assert local_result.fval < local_result.fval0


def test_petab_v2_schwen():
    problem_id = "Schwen_PONE2014"
    petab_problem = v2.Problem.from_yaml(
        models.get_problem_yaml_path(problem_id)
    )
    assert petab_problem.n_priors
    importer = PetabImporter(petab_problem)
    problem = importer.create_problem()
    assert problem.x_names == petab_problem.x_ids
    assert isinstance(problem.objective, pypesto.objective.AggregatedObjective)
    fval = problem.objective(np.asarray(petab_problem.x_nominal_free))
    assert np.isfinite(fval)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(PetabImportTest())
    unittest.main()
