"""
This is for testing the pypesto.Objective.
"""

import os

import amici
import numpy as np
import petab
import pytest

import pypesto
import pypesto.optimize as optimize
import pypesto.petab
from pypesto import C
from pypesto.objective.amici_util import add_sim_grad_to_opt_grad

from .petab_util import folder_base

ATOL = 1e-1
RTOL = 1e-0


def test_add_sim_grad_to_opt_grad():
    """
    Test gradient mapping/summation works as expected.
    17 = 1 + 2*5 + 2*3
    """
    par_opt_ids = ['opt_par_1', 'opt_par_2', 'opt_par_3']
    mapping_par_opt_to_par_sim = {
        'sim_par_1': 'opt_par_1',
        'sim_par_2': 'opt_par_3',
        'sim_par_3': 'opt_par_3',
    }
    par_sim_ids = ['sim_par_1', 'sim_par_2', 'sim_par_3']

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


def test_error_leastsquares_with_ssigma():
    petab_problem = petab.Problem.from_yaml(
        folder_base + "Zheng_PNAS2012/Zheng_PNAS2012.yaml"
    )
    petab_problem.model_name = "Zheng_PNAS2012"
    importer = pypesto.petab.PetabImporter(petab_problem)
    obj = importer.create_objective()
    problem = importer.create_problem(obj)

    optimizer = pypesto.optimize.ScipyOptimizer(
        'ls_trf', options={'max_nfev': 50}
    )
    with pytest.raises(RuntimeError):
        optimize.minimize(
            problem=problem,
            optimizer=optimizer,
            n_starts=1,
            filename=None,
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
        os.path.join(folder_base, model_name, model_name + '.yaml')
    )
    problem = importer.create_problem()
    obj = problem.objective
    obj.amici_solver.setNewtonMaxSteps(0)
    obj.amici_model.setSteadyStateSensitivityMode(
        amici.SteadyStateSensitivityMode.integrationOnly
    )
    obj.amici_solver.setAbsoluteTolerance(1e-12)
    obj.amici_solver.setRelativeTolerance(1e-12)

    # assert that initial guess is uninformative
    assert obj.steadystate_guesses['fval'] == np.inf

    optimizer = optimize.ScipyOptimizer()
    startpoints = pypesto.startpoint.UniformStartpoints(check_fval=False)

    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=1,
        startpoint_method=startpoints,
        filename=None,
        progress_bar=False,
    )

    assert obj.steadystate_guesses['fval'] < np.inf
    assert len(obj.steadystate_guesses['data']) == len(obj.edatas)
    # check that we have test a problem where plist is nontrivial
    assert any(len(e.plist) != len(e.parameters) for e in obj.edatas)

    df = obj.check_grad(
        problem.get_reduced_vector(
            result.optimize_result.list[0]['x'], problem.x_free_indices
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
    assert obj.steadystate_guesses['fval'] == np.inf
