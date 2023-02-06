from pathlib import Path
from typing import Dict, List

import numpy as np
import petab
import pytest

import pypesto
import pypesto.logging
import pypesto.optimize
import pypesto.petab
from pypesto.C import LIN, MAX, MODE_FUN, REDUCED, STANDARD, InnerParameterType
from pypesto.hierarchical.optimal_scaling import (
    OptimalScalingInnerSolver,
    OptimalScalingProblem,
)
from pypesto.hierarchical.optimal_scaling.parameter import (
    OptimalScalingParameter,
)
from pypesto.hierarchical.optimal_scaling.solver import (
    compute_interval_constraints,
    get_surrogate_all,
)

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


def test_optimal_scaling_calculator_and_objective():
    """Test the optimal scaling calculation of objective values and."""
    petab_problem = petab.Problem.from_yaml(example_ordinal_yaml)

    methods = [STANDARD, REDUCED]

    options_per_method = {
        STANDARD: {'method': STANDARD, 'reparameterized': False},
        REDUCED: {'method': REDUCED, 'reparameterized': False},
    }
    problems = {}

    for method, options in options_per_method.items():
        importer = pypesto.petab.PetabImporter(petab_problem, ordinal=True)
        objective = importer.create_objective(
            inner_solver_options=options,
        )
        problem = importer.create_problem(objective)
        problems[method] = problem

    def calculate(problem, x_dct):
        return problem.objective.calculator(
            x_dct=x_dct,
            sensi_orders=(0, 1),
            mode=MODE_FUN,
            amici_model=problem.objective.amici_model,
            amici_solver=problem.objective.amici_solver,
            edatas=problem.objective.edatas,
            n_threads=1,
            x_ids=petab_problem.x_ids,
            parameter_mapping=problem.objective.parameter_mapping,
            fim_for_hess=False,
        )

    x_dct = dict(zip(petab_problem.x_ids, petab_problem.x_nominal_scaled))

    calculator_results = {
        method: calculate(problems[method], x_dct=x_dct) for method in methods
    }

    # The results of the objective gradient and function value
    # should not depend on the method given.
    assert np.isclose(
        calculator_results[STANDARD]['fval'],
        calculator_results[REDUCED]['fval'],
    )
    assert np.isclose(
        calculator_results[STANDARD]['grad'],
        calculator_results[REDUCED]['grad'],
    ).all()

    # Since the nominal parameters are close to true ones, the
    # the fval and grad should both be low.
    assert np.all(calculator_results[STANDARD]['fval'] < 0.2)
    assert np.all(calculator_results[STANDARD]['grad'] < 0.1)


def _inner_problem_exp():
    timepoints = np.linspace(1, 10, 10)

    expected_inner_parameter_values = {
        STANDARD: np.asarray(
            [
                2.07281338e-14,
                1.99999999e00,
                2.58823529e00,
                4.00689557e00,
                4.59542030e00,
                6.23751538e00,
                6.82578436e00,
                8.40233862e00,
                8.99060060e00,
                1.10767572e01,
            ]
        ),
        REDUCED: np.asarray(
            [1.99999999, 4.00689557, 6.23751538, 8.40233862, 11.07675724]
        ),
    }

    simulation = timepoints
    data = np.full(simulation.shape, np.nan)

    par_types = ['cat_lb', 'cat_ub']

    categories = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    inner_parameter_ids = [
        f'{par_type}_{category}'
        for par_type in par_types
        for category in set(categories)
    ]

    # Get data masks for inner parameters
    masks = []
    for category in categories:
        new_mask = np.full(data.shape, False)
        new_mask[2 * (category - 1) : 2 * category] = True
        masks.append([new_mask])

    # Construct inner parameters
    inner_parameters = [
        OptimalScalingParameter(
            inner_parameter_id=inner_parameter_id,
            inner_parameter_type=InnerParameterType.OPTIMAL_SCALING,
            scale=LIN,
            lb=-np.inf,
            ub=np.inf,
            ixs=mask,
            group=1,
            category=inner_parameter_category,
        )
        for inner_parameter_id, mask, inner_parameter_category in zip(
            inner_parameter_ids, masks, categories
        )
    ]

    # Construct inner problem
    inner_problem = OptimalScalingProblem(
        xs=inner_parameters, data=[data], method=STANDARD
    )

    return inner_problem, expected_inner_parameter_values, simulation


def test_optimal_scaling_solver():
    """Test the Optimal scaling solver."""
    (
        inner_problem,
        expected_values,
        simulation,
    ) = _inner_problem_exp()

    rtol = 1e-3

    solver = OptimalScalingInnerSolver(
        options={'method': STANDARD, 'reparameterized': False}
    )

    standard_result = solver.solve(
        problem=inner_problem,
        sim=[simulation],
    )[0]

    assert np.all(
        np.isclose(standard_result['x'], expected_values[STANDARD], rtol=rtol)
    )
    assert np.all(np.isclose(standard_result['fun'], 0, rtol=rtol))
    assert np.all(np.isclose(standard_result['jac'], 0, rtol=rtol))

    solver = OptimalScalingInnerSolver(
        options={'method': REDUCED, 'reparameterized': False}
    )

    reduced_result = solver.solve(
        problem=inner_problem,
        sim=[simulation],
    )[0]

    assert np.all(
        np.isclose(reduced_result['x'], expected_values[REDUCED], rtol=rtol)
    )
    assert np.all(np.isclose(reduced_result['fun'], 0, rtol=rtol))
    assert np.all(np.isclose(reduced_result['jac'], 0, rtol=rtol))


def test_surrogate_data_analytical_calculation():
    """Test analytical calculation of surrogate data.

    We test the analytical calculation of the surrogate data with respect
    to the simulation and category bounds.
    """
    inner_problem, inner_parameters, sim = _inner_problem_exp()

    rtol = 1e-3

    optimal_inner_parameters = inner_parameters[STANDARD]

    expected_values = {
        'interval_range': 0.9090909090909091,
        'interval_gap': 0.5882352941176472,
        'surrogate_data': np.linspace(1, 10, 10),
    }

    options = {
        'method': STANDARD,
        'reparameterized': False,
        'intervalConstraints': MAX,
        'minGap': 1e-16,
    }

    category_upper_bounds = inner_problem.get_cat_ub_parameters_for_group(1)

    interval_range, interval_gap = compute_interval_constraints(
        category_upper_bounds, [sim], options
    )

    surrogate_data, _, _ = get_surrogate_all(
        category_upper_bounds,
        optimal_inner_parameters,
        [sim],
        interval_range,
        interval_gap,
        options,
    )

    assert np.isclose(
        interval_range, expected_values['interval_range'], rtol=rtol
    )
    assert np.isclose(interval_gap, expected_values['interval_gap'], rtol=rtol)
    assert np.all(
        np.isclose(
            surrogate_data, expected_values['surrogate_data'], rtol=rtol
        )
    )
