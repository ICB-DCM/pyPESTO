from pathlib import Path
from typing import Dict, List

import numpy as np
import petab
import pytest

import pypesto
import pypesto.logging
import pypesto.optimize
import pypesto.petab

inner_solver_options = [
    [
        {
            key: value
            for key, value in zip(
                ['method', 'reparameterized', 'intervalConstraints'],
                [method, reparameterized, intervalConstraints],
            )
        }
        for method, reparameterized in zip(
            ['standard', 'reduced', 'reduced'], [False, False, True]
        )
    ]
    for intervalConstraints in ['max', 'max-min']
]

example_ordinal_yaml = (
    Path(__file__).parent
    / '..'
    / '..'
    / 'doc'
    / 'example'
    / 'example_ordinal'
    / 'example_ordinal.yaml'
)


@pytest.fixture(params=inner_solver_options)
def inner_solver_options(request):
    return request.param


def test_evaluate_objective(inner_solver_options: List[Dict]):
    """Check that standard / reduced / reparameterized formulations yield the
    same result."""
    petab_problem = petab.Problem.from_yaml(example_ordinal_yaml)
    vals = []
    for idx, option in enumerate(inner_solver_options):
        problem = _create_problem(petab_problem, option)
        val = problem.objective(np.array([0, 0]))
        vals.append(val)
        assert np.isclose(vals[idx], vals[idx - 1])


def test_optimization(inner_solver_options: List[Dict]):
    """Check that optimizations finishes without error."""
    petab_problem = petab.Problem.from_yaml(example_ordinal_yaml)

    optimizer = pypesto.optimize.ScipyOptimizer(
        method='Nelder-Mead', options={'maxiter': 10}
    )
    for option in inner_solver_options:
        problem = _create_problem(petab_problem, option)
        pypesto.optimize.minimize(
            problem=problem, n_starts=1, optimizer=optimizer
        )


def _create_problem(
    petab_problem: petab.Problem, option: Dict
) -> pypesto.Problem:
    """Creates the ordinal pyPESTO problem with given options."""
    importer = pypesto.petab.PetabImporter(petab_problem, ordinal=True)
    importer.create_model()

    objective = importer.create_objective(
        inner_solver_options=option,
    )
    problem = importer.create_problem(objective)
    return problem
