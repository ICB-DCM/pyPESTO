"""
This is for testing the pypesto.Objective.
"""

import os

import amici
import benchmark_models_petab as models
import numpy as np
import petab.v1 as petab
import pytest

import pypesto
import pypesto.optimize as optimize
import pypesto.petab
from pypesto import C
from pypesto.objective.amici.amici_util import add_sim_grad_to_opt_grad

ATOL = 1e-1
RTOL = 1e-0


def test_add_sim_grad_to_opt_grad():
    """
    Test gradient mapping/summation works as expected.
    17 = 1 + 2*5 + 2*3
    """
    par_opt_ids = ["opt_par_1", "opt_par_2", "opt_par_3"]
    mapping_par_opt_to_par_sim = {
        "sim_par_1": "opt_par_1",
        "sim_par_2": "opt_par_3",
        "sim_par_3": "opt_par_3",
    }
    par_sim_ids = ["sim_par_1", "sim_par_2", "sim_par_3"]

    sim_grad = np.asarray([1.0, 3.0, 5.0])
    opt_grad = np.asarray([1.0, 1.0, 1.0])
    expected = np.asarray([3.0, 1.0, 17.0])

    add_sim_grad_to_opt_grad(
        par_opt_ids,
        par_sim_ids,
        mapping_par_opt_to_par_sim,
        sim_grad,
        opt_grad,
        coefficient=2.0,
    )

    assert np.allclose(expected, opt_grad)


@pytest.mark.flaky(reruns=2)
def test_error_leastsquares_with_ssigma():
    model_name = "Zheng_PNAS2012"
    petab_problem = petab.Problem.from_yaml(
        os.path.join(models.MODELS_DIR, model_name, model_name + ".yaml")
    )
    petab_problem.model_name = model_name
    importer = pypesto.petab.PetabImporter(petab_problem)
    obj = importer.create_objective_creator().create_objective()
    problem = importer.create_problem(
        obj, startpoint_kwargs={"check_fval": True, "check_grad": True}
    )

    optimizer = pypesto.optimize.ScipyOptimizer(
        "ls_trf", options={"max_nfev": 50}
    )
    with pytest.raises(RuntimeError):
        optimize.minimize(
            problem=problem,
            optimizer=optimizer,
            n_starts=1,
            options=optimize.OptimizeOptions(allow_failed_starts=False),
            progress_bar=False,
        )


@pytest.mark.flaky(reruns=5)
def test_preeq_guesses():
    """
    Test whether optimization with preequilibration guesses works, asserts
    that steadystate guesses are written and checks that gradient is still
    correct with guesses set.
    """
    model_name = "Brannmark_JBC2010"
    importer = pypesto.petab.PetabImporter.from_yaml(
        os.path.join(models.MODELS_DIR, model_name, model_name + ".yaml")
    )
    obj_creator = importer.create_objective_creator()
    amici_model = obj_creator.create_model()
    amici_model.set_steady_state_computation_mode(
        amici.SteadyStateComputationMode.integrateIfNewtonFails
    )
    amici_model.set_steady_state_sensitivity_mode(
        amici.SteadyStateSensitivityMode.integrateIfNewtonFails
    )
    obj = obj_creator.create_objective(model=amici_model)
    problem = importer.create_problem(objective=obj)
    obj.amici_model.set_steady_state_sensitivity_mode(
        amici.SteadyStateSensitivityMode.integrationOnly
    )
    obj = problem.objective
    obj.amici_solver.set_newton_max_steps(0)
    obj.amici_solver.set_absolute_tolerance(1e-12)
    obj.amici_solver.set_relative_tolerance(1e-12)

    # assert that initial guess is uninformative
    assert obj.steadystate_guesses["fval"] == np.inf

    optimizer = optimize.ScipyOptimizer()
    problem.startpoint_method = pypesto.startpoint.UniformStartpoints(
        check_fval=False
    )

    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=1,
        progress_bar=False,
    )

    assert obj.steadystate_guesses["fval"] < np.inf
    assert len(obj.steadystate_guesses["data"]) == len(obj.edatas)
    # check that we have test a problem where plist is nontrivial
    assert any(len(e.plist) != len(e.parameters) for e in obj.edatas)

    df = obj.check_grad(
        problem.get_reduced_vector(
            result.optimize_result.list[0]["x"], problem.x_free_indices
        ),
        eps=1e-3,
        verbosity=0,
        mode=C.MODE_FUN,
    )
    print("relative errors MODE_FUN: ", df.rel_err.values)
    print("absolute errors MODE_FUN: ", df.abs_err.values)
    assert np.all((df.rel_err.values < RTOL) | (df.abs_err.values < ATOL))

    # assert that resetting works
    problem.objective.initialize()
    assert obj.steadystate_guesses["fval"] == np.inf
