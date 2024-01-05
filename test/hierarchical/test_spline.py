from pathlib import Path
from typing import Dict

import numpy as np
import petab
import pytest

import pypesto
import pypesto.logging
import pypesto.optimize
import pypesto.petab
from pypesto.C import (
    INNER_NOISE_PARS,
    LIN,
    MODE_FUN,
    OPTIMIZE_NOISE,
    InnerParameterType,
)
from pypesto.hierarchical.semiquantitative import (
    SemiquantInnerSolver,
    SemiquantProblem,
)
from pypesto.hierarchical.semiquantitative.parameter import (
    SplineInnerParameter,
)
from pypesto.hierarchical.semiquantitative.solver import (
    _calculate_nllh_for_group,
    _calculate_regularization_for_group,
    _calculate_regularization_gradient_for_group,
    _calculate_sigma_for_group,
    extract_expdata_using_mask,
    get_monotonicity_measure,
    get_spline_mapped_simulations,
)

inner_options = [
    {
        'spline_ratio': spline_ratio,
        'min_diff_factor': min_diff_factor,
        'regularize_spline': regularize_spline,
        'regularization_factor': regularization_factor,
    }
    for spline_ratio in [1.0, 1 / 4]
    for min_diff_factor in [1 / 2, 0.0]
    for regularize_spline in [True, False]
    for regularization_factor in [1.0, 0.0]
]

example_semiquantitative_yaml = (
    Path(__file__).parent
    / '..'
    / '..'
    / 'doc'
    / 'example'
    / 'example_semiquantitative'
    / 'example_semiquantitative_linear.yaml'
)


@pytest.fixture(params=inner_options)
def inner_options(request):
    return request.param


def test_optimization(inner_options: Dict):
    """Check that optimizations finishes without error."""
    petab_problem = petab.Problem.from_yaml(example_semiquantitative_yaml)
    # Set seed for reproducibility.
    np.random.seed(0)
    optimizer = pypesto.optimize.ScipyOptimizer(
        method="L-BFGS-B",
        options={"disp": None, "ftol": 2.220446049250313e-09, "gtol": 1e-5},
    )
    problem = _create_problem(petab_problem, inner_options)
    result = pypesto.optimize.minimize(
        problem=problem, n_starts=1, optimizer=optimizer
    )
    # Check that optimization finished without infinite or nan values.
    assert np.isfinite(result.optimize_result.list[0]['fval'])
    assert np.all(np.isfinite(result.optimize_result.list[0]['x']))
    assert np.all(np.isfinite(result.optimize_result.list[0]['grad'][2:]))
    # Check that optimization finished with a lower value.
    assert (
        result.optimize_result.list[0]['fval']
        < result.optimize_result.list[0]['fval0']
    )


def _create_problem(
    petab_problem: petab.Problem, option: Dict
) -> pypesto.Problem:
    """Creates the spline pyPESTO problem with given options."""
    importer = pypesto.petab.PetabImporter(
        petab_problem,
        hierarchical=True,
    )
    importer.create_model()

    objective = importer.create_objective(
        inner_options=option,
    )
    problem = importer.create_problem(objective)
    return problem


def test_spline_calculator_and_objective():
    """Test the spline calculation of objective values."""
    petab_problem = petab.Problem.from_yaml(example_semiquantitative_yaml)

    problems = {}
    options = {
        'minimal_diff_on': {
            'spline_ratio': 1 / 2,
            'min_diff_factor': 1 / 2,
        },
        'minimal_diff_off': {
            'spline_ratio': 1 / 2,
            'min_diff_factor': 0.0,
        },
    }

    for minimal_diff, option in options.items():
        importer = pypesto.petab.PetabImporter(
            petab_problem,
            hierarchical=True,
        )
        objective = importer.create_objective(
            inner_options=option,
        )
        problem = importer.create_problem(objective)
        problems[minimal_diff] = problem

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

    def inner_calculate(problem, x_dct):
        return problem.objective.calculator.inner_calculators[0](
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
        minimal_diff: calculate(problems[minimal_diff], x_dct=x_dct)
        for minimal_diff in options.keys()
    }
    inner_calculator_results = {
        minimal_diff: inner_calculate(problems[minimal_diff], x_dct=x_dct)
        for minimal_diff in options.keys()
    }

    finite_differences = pypesto.objective.FD(problem.objective)
    FD_results = finite_differences(
        x=petab_problem.x_nominal_scaled,
        sensi_orders=(0, 1),
        mode=MODE_FUN,
    )

    atol = 1e-3
    grad_atol = 1e-2

    # Check the inner calculator and the inner calculator collector
    # give the same results.
    assert np.allclose(
        inner_calculator_results['minimal_diff_on']['fval'],
        calculator_results['minimal_diff_on']['fval'],
        atol=atol,
    )
    assert np.allclose(
        inner_calculator_results['minimal_diff_on']['grad'],
        calculator_results['minimal_diff_on']['grad'],
        atol=atol,
    )

    # For nominal parameters, the objective function and gradient
    # will not depend on whether we constrain minimal difference.
    # In general, this is not the case.
    assert np.isclose(
        calculator_results['minimal_diff_on']['fval'],
        calculator_results['minimal_diff_off']['fval'],
        atol=atol,
    )
    assert np.allclose(
        calculator_results['minimal_diff_on']['grad'],
        calculator_results['minimal_diff_off']['grad'],
        atol=atol,
    )

    # The gradient should be close to the one calculated using
    # finite differences.
    assert np.allclose(
        calculator_results['minimal_diff_on']['grad'],
        FD_results[1],
        atol=atol,
    )

    # Since the nominal parameters are close to true ones, the
    # the fval and grad should both be low.
    expected_fval = np.log(2 * np.pi) * 18 / 2
    assert np.isclose(
        calculator_results['minimal_diff_on']['fval'], expected_fval, atol=atol
    )
    assert np.all(calculator_results['minimal_diff_off']['grad'] < grad_atol)


def test_extract_expdata_using_mask():
    """Test the extraction of expdata using a mask."""
    expdata = [
        np.array([1, 2, 3, 4, 5]),
        np.array([6, 7, 8, 9, 10]),
    ]
    mask = [
        np.array([True, False, True, False, True]),
        np.array([False, True, False, True, False]),
    ]
    assert np.all(
        extract_expdata_using_mask(expdata, mask) == np.array([1, 3, 5, 7, 9])
    )


def test_get_monotonicity_measure():
    """Test the calculation of the monotonicity measure."""
    measurement = np.array([1, 2, 3, 4, 5])
    simulation = np.array([1, 2, 3, 4, 5])
    assert get_monotonicity_measure(measurement, simulation) == 0

    measurement = np.array([1, 2, 3, 4, 5])
    simulation = np.array([5, 4, 3, 2, 1])
    assert get_monotonicity_measure(measurement, simulation) == 10


def _inner_problem_exp():
    n_timepoints = 11
    timepoints = np.linspace(0, 10, n_timepoints)

    simulation = timepoints
    sigma = np.full(len(timepoints), 1)
    data = timepoints

    spline_ratio = 1 / 2
    n_spline_pars = int(np.ceil(spline_ratio * len(timepoints)))

    expected_values = {
        'fun': np.log(2 * np.pi) * n_timepoints / 2,
        'jac': np.zeros(n_spline_pars),
        'x': np.asarray([0.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    }

    par_type = 'spline'
    mask = [np.full(len(simulation), True)]

    inner_parameters = [
        SplineInnerParameter(
            inner_parameter_id=f'{par_type}_{1}_{par_index+1}',
            inner_parameter_type=InnerParameterType.SPLINE,
            scale=LIN,
            lb=-np.inf,
            ub=np.inf,
            observable_id='obs1',
            ixs=mask,
            index=par_index + 1,
            group=1,
        )
        for par_index in range(n_spline_pars)
    ]

    inner_problem = SemiquantProblem(
        xs=inner_parameters,
        data=[data],
        edatas=None,
        spline_ratio=spline_ratio,
    )

    return inner_problem, expected_values, simulation, sigma


def test_spline_inner_solver():
    """Test the spline inner solver."""
    inner_problem, expected_values, simulation, sigma = _inner_problem_exp()

    options = {
        'minimal_diff_on': {
            'min_diff_factor': 1 / 2,
        },
        'minimal_diff_off': {
            'min_diff_factor': 0.0,
        },
    }

    rtol = 1e-6

    inner_solvers = {}
    results = {}

    for minimal_diff, option in options.items():
        inner_solvers[minimal_diff] = SemiquantInnerSolver(
            options=option,
        )

        results[minimal_diff] = inner_solvers[minimal_diff].solve(
            problem=inner_problem,
            sim=[simulation],
            amici_sigma=[sigma],
        )

    for minimal_diff in options.keys():
        assert np.isclose(
            results[minimal_diff][0]['fun'], expected_values['fun'], rtol=rtol
        )
        assert np.allclose(
            results[minimal_diff][0]['jac'], expected_values['jac'], rtol=rtol
        )
        assert np.allclose(
            results[minimal_diff][0]['x'], expected_values['x'], rtol=rtol
        )


def test_get_spline_mapped_simulations():
    """Test the mapping of model simulations using the spline."""
    spline_parameters = np.array([2, 4, 6, 8, 15])
    simulation = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.2, 5])
    n_spline_pars = 5
    distance_between_bases = 1
    spline_bases = np.array([1, 2, 3, 4, 5])
    simulation_intervals = (
        np.ceil((simulation - spline_bases[0]) / distance_between_bases) + 1
    ).astype(int)

    rtol = 1e-6

    expected_spline_mapped_simulations = np.array(
        [2, 4, 6, 9, 12, 16, 20, 23, 35]
    )

    spline_mapped_simulations = get_spline_mapped_simulations(
        spline_parameters,
        simulation,
        n_spline_pars,
        distance_between_bases,
        spline_bases,
        simulation_intervals,
    )
    assert np.allclose(
        spline_mapped_simulations,
        expected_spline_mapped_simulations,
        rtol=rtol,
    )


def test_calculate_sigma_for_group():
    """Test the calculation of sigma for a group."""
    expected_sigma = np.sqrt(2 * 12.0 / 8)
    residuals_squared = 12

    sigma = _calculate_sigma_for_group(residuals_squared, n_datapoints=8)
    assert sigma == expected_sigma


def test_calculate_nllh_for_group():
    """Test the calculation of the nllh for a group."""
    n_timepoints = 11
    timepoints = np.linspace(0, 10, n_timepoints)

    simulation = timepoints
    data = timepoints

    spline_parameters = np.asarray([0.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    spline_ratio = 1 / 2
    n_spline_pars = int(np.ceil(spline_ratio * len(timepoints)))

    spline_base_distance = 2.0
    spline_bases = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    simulation_intervals = [
        int(np.ceil((sim - spline_bases[0]) / spline_base_distance)) + 1
        for sim in simulation
    ]

    group_dict = {
        OPTIMIZE_NOISE: False,
        INNER_NOISE_PARS: 1,
    }

    expected_nllh = np.log(2 * np.pi) * n_timepoints / 2

    nllh = _calculate_nllh_for_group(
        spline_parameters,
        simulation,
        data,
        n_spline_pars,
        spline_base_distance,
        spline_bases,
        simulation_intervals,
        regularization_factor=0.0,
        regularize_spline=False,
        group_dict=group_dict,
    )
    assert nllh == expected_nllh


def test_calculate_regularization_for_group():
    """Test the calculation of the regularization for a group."""
    spline_parameters = np.array([2, 1, 1, 1, 0, 2])
    n_spline_parameters = len(spline_parameters)
    spline_bases = np.array([1, 2, 3, 4, 5, 5])
    regularization_factor = 1.0

    lower_trian = np.tril(np.ones((n_spline_parameters, n_spline_parameters)))
    xi = np.dot(lower_trian, spline_parameters)

    expected_beta = 1
    expected_alpha = 1

    expected_regularization = (
        regularization_factor
        * np.sum((xi - expected_alpha * spline_bases - expected_beta) ** 2)
        / (2 * n_spline_parameters)
    )

    expected_regularization_gradient = (
        regularization_factor
        * np.dot(
            xi - expected_alpha * spline_bases - expected_beta,
            lower_trian,
        )
        / n_spline_parameters
    )

    regularization = _calculate_regularization_for_group(
        spline_parameters,
        n_spline_parameters,
        spline_bases,
        regularization_factor,
    )
    regularization_gradient = _calculate_regularization_gradient_for_group(
        spline_parameters,
        n_spline_parameters,
        spline_bases,
        regularization_factor,
    )

    assert np.isclose(regularization, expected_regularization)
    assert np.allclose(
        regularization_gradient,
        expected_regularization_gradient,
    )
