from pathlib import Path
from typing import Dict, List

import numpy as np
import petab
import pytest

import pypesto
import pypesto.logging
import pypesto.optimize
import pypesto.petab
from pypesto.hierarchical.optimal_scaling_approach.optimal_scaling_solver import (
    OptimalScalingInnerSolver,
)

inner_solver_options = [
    [
        {
            'method': 'standard',
            'reparameterized': False,
            'intervalConstraints': 'max',
        },
        {
            'method': 'reduced',
            'reparameterized': False,
            'intervalConstraints': 'max',
        },
        {
            'method': 'reduced',
            'reparameterized': True,
            'intervalConstraints': 'max',
        },
    ],
    [
        {
            'method': 'standard',
            'reparameterized': False,
            'intervalConstraints': 'max-min',
        },
        {
            'method': 'reduced',
            'reparameterized': False,
            'intervalConstraints': 'max-min',
        },
        {
            'method': 'reduced',
            'reparameterized': True,
            'intervalConstraints': 'max-min',
        },
    ],
]

example_qualitative_yaml = (
    Path(__file__).parent
    / '..'
    / '..'
    / 'doc'
    / 'example'
    / 'example_qualitative'
    / 'example_qualitative.yaml'
)


@pytest.fixture(params=inner_solver_options)
def inner_solver_options(request):
    return request.param


def test_evaluate_objective(inner_solver_options: List[Dict]):
    """Check that standard / reduced / reparameterized formulations yield the
    same result."""
    petab_problem = petab.Problem.from_yaml(example_qualitative_yaml)
    vals = []
    for idx, option in enumerate(inner_solver_options):
        problem = create_problem(petab_problem, option)
        val = problem.objective(np.array([0, 0]))
        vals.append(val)
        assert np.isclose(vals[idx], vals[idx - 1])


def test_optimization(inner_solver_options: List[Dict]):
    """Check that optimizations finishes without error."""
    petab_problem = petab.Problem.from_yaml(example_qualitative_yaml)

    optimizer = pypesto.optimize.ScipyOptimizer(
        method='Nelder-Mead', options={'maxiter': 10}
    )
    for option in inner_solver_options:
        problem = create_problem(petab_problem, option)
        pypesto.optimize.minimize(
            problem=problem, n_starts=1, optimizer=optimizer
        )


def create_problem(
    petab_problem: petab.Problem, option: Dict
) -> pypesto.Problem:
    importer = pypesto.petab.PetabImporter(petab_problem, ordinal=True)
    importer.create_model()

    objective = importer.create_objective(
        inner_problem_method=option['method'],
        inner_solver_options=option,
    )
    problem = importer.create_problem(objective)
    return problem
