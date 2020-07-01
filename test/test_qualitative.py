import pypesto
import pypesto.petab
import pypesto.optimize
from pypesto.hierarchical.optimal_scaling_solver import OptimalScalingInnerSolver
import pypesto.logging

import pytest
import os
import amici
import petab
import numpy as np
import logging

inner_problem_options = [
    [{'method': 'standard',
      'reparameterized': False,
      'intervalConstraints': 'max'},
     {'method': 'reduced',
      'reparameterized': False,
      'intervalConstraints': 'max'},
     {'method': 'reduced',
      'reparameterized': True,
      'intervalConstraints': 'max'}],
    [{'method': 'standard',
      'reparameterized': False,
      'intervalConstraints': 'max-min'},
     {'method': 'reduced',
      'reparameterized': False,
      'intervalConstraints': 'max-min'},
     {'method': 'reduced',
      'reparameterized': True,
      'intervalConstraints': 'max-min'}]
]


@pytest.fixture(params=inner_problem_options)
def inner_problem_option(request):
    return request.param


def test_evaluate_objective(inner_problem_option):
    folder_base = 'doc/example/'
    model_name = 'example_qualitative'
    yaml_config = os.path.join(folder_base, model_name,
                               model_name + '.yaml')
    petab_problem = petab.Problem.from_yaml(yaml_config)
    vals = []
    for idx, option in enumerate(inner_problem_option):
        problem = create_problem(petab_problem, option)
        val = problem.objective(np.array([0, 0]))
        vals.append(val)
        assert np.isclose(vals[idx], vals[idx-1])


def test_optimization(inner_problem_option):
    folder_base = 'doc/example/'
    model_name = 'example_qualitative'
    yaml_config = os.path.join(folder_base, model_name,
                               model_name + '.yaml')
    petab_problem = petab.Problem.from_yaml(yaml_config)

    optimizer = pypesto.optimize.ScipyOptimizer(method='Nelder-Mead', options={'maxiter': 10})
    for option in inner_problem_option:
        problem = create_problem(petab_problem, option)
        pypesto.optimize.minimize(problem=problem, n_starts=1, optimizer=optimizer)


def create_problem(petab_problem, option):
    importer = pypesto.petab.PetabImporter(petab_problem)

    model = importer.create_model()

    objective = importer.create_objective(hierarchical=True)
    problem = importer.create_problem(objective)
    problem.objective.calculator.inner_solver = OptimalScalingInnerSolver(options=option)
    return problem

