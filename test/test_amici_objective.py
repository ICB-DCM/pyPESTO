"""
This is for testing the pypesto.Objective.
"""

from pypesto.objective.amici_util import add_sim_grad_to_opt_grad

import petab
import pypesto
import pypesto.optimize
import pypesto.objective.constants
import pytest
import numpy as np
from test.petab_util import folder_base

ATOL = 1e-1
RTOL = 1e-0


def test_add_sim_grad_to_opt_grad():
    """
    Test gradient mapping/summation works as expected.
    """
    par_opt_ids = ['opt_par_1',
                   'opt_par_2',
                   'opt_par_3']
    mapping_par_opt_to_par_sim = {
        'sim_par_1': 'opt_par_1',
        'sim_par_2': 'opt_par_3',
        'sim_par_3': 'opt_par_3'
    }
    par_sim_ids = ['sim_par_1', 'sim_par_2', 'sim_par_3']

    sim_grad = [1.0, 3.0, 5.0]
    opt_grad = [1.0, 1.0, 1.0]
    expected = [3.0, 1.0, 17.0]

    add_sim_grad_to_opt_grad(
        par_opt_ids,
        par_sim_ids,
        mapping_par_opt_to_par_sim,
        sim_grad,
        opt_grad,
        coefficient=2.0)

    assert expected == opt_grad


def test_error_leastsquares_with_ssigma():
    petab_problem = petab.Problem.from_yaml(
        folder_base + "Zheng_PNAS2012/Zheng_PNAS2012.yaml")
    petab_problem.model_name = "Zheng_PNAS2012"
    importer = pypesto.petab.PetabImporter(petab_problem)
    obj = importer.create_objective()
    problem = importer.create_problem(obj)

    optimizer = pypesto.optimize.ScipyOptimizer(
        'ls_trf', options={'max_nfev': 50})
    with pytest.raises(RuntimeError):
        pypesto.optimize.minimize(
            problem=problem, optimizer=optimizer, n_starts=1,
            options=pypesto.optimize.OptimizeOptions(allow_failed_starts=False)
        )


def test_preeq_guesses():
    """
    Test whether optimization with preequilibration guesses works, asserts
    that steadystate guesses are written and checks that gradient is still
    correct with guesses set.
    """
    petab_problem = petab.Problem.from_yaml(
        folder_base + "Zheng_PNAS2012/Zheng_PNAS2012.yaml")
    petab_problem.model_name = "Zheng_PNAS2012"
    importer = pypesto.petab.PetabImporter(petab_problem)
    obj = importer.create_objective()
    problem = importer.create_problem(obj)

    # assert that initial guess is uninformative
    assert problem.objective.steadystate_guesses['fval'] == np.inf

    optimizer = pypesto.optimize.ScipyOptimizer(
        'L-BFGS-B', options={'maxiter': 50})
    result = pypesto.optimize.minimize(
        problem=problem, optimizer=optimizer, n_starts=1,
    )

    assert problem.objective.steadystate_guesses['fval'] < np.inf
    assert len(obj.steadystate_guesses['data']) == 1

    df = obj.check_grad(
        result.optimize_result.list[0]['x'],
        eps=1e-3,
        verbosity=0,
        mode=pypesto.objective.constants.MODE_FUN
    )
    print("relative errors MODE_FUN: ", df.rel_err.values)
    print("absolute errors MODE_FUN: ", df.abs_err.values)
    assert np.all((df.rel_err.values < RTOL) | (df.abs_err.values < ATOL))

    # assert that resetting works
    problem.objective.initialize()
    assert problem.objective.steadystate_guesses['fval'] == np.inf
