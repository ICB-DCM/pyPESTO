from pathlib import Path

import numpy as np
import petab
import pytest

import pypesto
import pypesto.logging
import pypesto.optimize
import pypesto.petab
from pypesto.hierarchical.optimal_scaling_solver import (
    OptimalScalingInnerSolver,
)

inner_problem_options = [
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


@pytest.fixture(params=inner_problem_options)
def inner_problem_option(request):
    return request.param


def test_evaluate_objective(inner_problem_option):
    petab_problem = petab.Problem.from_yaml(example_qualitative_yaml)
    vals = []
    for idx, option in enumerate(inner_problem_option):
        problem = create_problem(petab_problem, option)
        val = problem.objective(np.array([0, 0]))
        vals.append(val)
        assert np.isclose(vals[idx], vals[idx - 1])


def test_optimization(inner_problem_option):
    petab_problem = petab.Problem.from_yaml(example_qualitative_yaml)

    optimizer = pypesto.optimize.ScipyOptimizer(
        method='Nelder-Mead', options={'maxiter': 10}
    )
    for option in inner_problem_option:
        problem = create_problem(petab_problem, option)
        pypesto.optimize.minimize(
            problem=problem, n_starts=1, optimizer=optimizer
        )


def create_problem(petab_problem, option):
    importer = pypesto.petab.PetabImporter(petab_problem)
    importer.create_model()

    objective = importer.create_objective(hierarchical=True)
    problem = importer.create_problem(objective)
    problem.objective.calculator.inner_solver = OptimalScalingInnerSolver(
        options=option
    )
    return problem
