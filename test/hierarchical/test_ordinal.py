from pathlib import Path

import numpy as np
import petab.v1 as petab
import pytest

import pypesto
import pypesto.logging
import pypesto.optimize
import pypesto.petab
from pypesto.C import (
    CAT_LB,
    CAT_UB,
    INTERVAL_CONSTRAINTS,
    LIN,
    MAX,
    MAXMIN,
    METHOD,
    MIN_GAP,
    MODE_FUN,
    REDUCED,
    REPARAMETERIZED,
    STANDARD,
    InnerParameterType,
)
from pypesto.hierarchical.ordinal import OrdinalInnerSolver, OrdinalProblem
from pypesto.hierarchical.ordinal.parameter import OrdinalParameter
from pypesto.hierarchical.ordinal.solver import (
    compute_interval_constraints,
    get_surrogate_all,
)

inner_options = [
    [
        dict(
            zip(
                [METHOD, REPARAMETERIZED, INTERVAL_CONSTRAINTS],
                [method, reparameterized, interval_constraints],
            )
        )
        for method, reparameterized in zip(
            [STANDARD, REDUCED, REDUCED], [False, False, True]
        )
    ]
    for interval_constraints in [MAX, MAXMIN]
]

example_ordinal_yaml = (
    Path(__file__).parent
    / ".."
    / ".."
    / "doc"
    / "example"
    / "example_ordinal"
    / "example_ordinal.yaml"
)


@pytest.fixture(params=inner_options)
def inner_options(request):
    return request.param


def test_evaluate_objective(inner_options: list[dict]):
    """Check that standard / reduced / reparameterized formulations yield the
    same result."""
    petab_problem = petab.Problem.from_yaml(example_ordinal_yaml)
    vals = []
    for idx, option in enumerate(inner_options):
        problem = _create_problem(petab_problem, option)
        val = problem.objective(np.array([0, 0]))
        vals.append(val)
        assert np.isclose(vals[idx], vals[idx - 1])


def test_optimization(inner_options: list[dict]):
    """Check that optimizations finishes without error."""
    petab_problem = petab.Problem.from_yaml(example_ordinal_yaml)
    # Set seed for reproducibility.
    np.random.seed(0)
    optimizer = pypesto.optimize.ScipyOptimizer(
        method="L-BFGS-B", options={"maxiter": 10}
    )
    for option in inner_options:
        problem = _create_problem(petab_problem, option)
        result = pypesto.optimize.minimize(
            problem=problem, n_starts=1, optimizer=optimizer
        )
        # Check that optimization finished without infinite or nan values.
        assert np.isfinite(result.optimize_result.list[0]["fval"])
        assert np.all(np.isfinite(result.optimize_result.list[0]["x"]))
        assert np.all(np.isfinite(result.optimize_result.list[0]["grad"][2:]))
        # Check that optimization finished with a lower objective value.
        assert (
            result.optimize_result.list[0]["fval"]
            < result.optimize_result.list[0]["fval0"]
        )


def _create_problem(
    petab_problem: petab.Problem, option: dict
) -> pypesto.Problem:
    """Creates the ordinal pyPESTO problem with given options."""
    importer = pypesto.petab.PetabImporter(petab_problem, hierarchical=True)
    factory = importer.create_factory()
    factory.create_model()

    objective = factory.create_objective(
        inner_options=option,
    )
    problem = importer.create_problem(objective)
    return problem


def test_ordinal_calculator_and_objective():
    """Test the ordinal calculation of objective and gradient values."""
    petab_problem = petab.Problem.from_yaml(example_ordinal_yaml)

    methods = [STANDARD, REDUCED]

    options_per_method = {
        STANDARD: {METHOD: STANDARD, REPARAMETERIZED: False},
        REDUCED: {METHOD: REDUCED, REPARAMETERIZED: False},
    }
    problems = {}

    for method, options in options_per_method.items():
        importer = pypesto.petab.PetabImporter(
            petab_problem, hierarchical=True
        )
        factory = importer.create_factory()
        objective = factory.create_objective(
            inner_options=options,
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
        method: calculate(problems[method], x_dct=x_dct) for method in methods
    }
    inner_calculator_results = {
        method: inner_calculate(problems[method], x_dct=x_dct)
        for method in methods
    }

    finite_differences = pypesto.objective.FD(
        problem.objective,
    )
    finite_differences_results = finite_differences(
        petab_problem.x_nominal_scaled,
        (
            0,
            1,
        ),
        MODE_FUN,
    )

    # Check the inner calculator and the inner calculator collector
    # give the same results.
    assert np.allclose(
        inner_calculator_results[STANDARD]["fval"],
        calculator_results[STANDARD]["fval"],
    )
    assert np.allclose(
        inner_calculator_results[STANDARD]["grad"],
        calculator_results[STANDARD]["grad"],
    )

    # The results of the objective gradient and function value
    # should not depend on the method given.
    assert np.isclose(
        calculator_results[STANDARD]["fval"],
        calculator_results[REDUCED]["fval"],
    )
    assert np.allclose(
        calculator_results[STANDARD]["grad"],
        calculator_results[REDUCED]["grad"],
    )

    # Check that the gradient is the same as the one obtained
    # with finite differences.
    assert np.allclose(
        finite_differences_results[1],
        calculator_results[STANDARD]["grad"],
    )

    # Since the nominal parameters are close to true ones,
    # the fval and grad should both be low.
    assert np.all(calculator_results[STANDARD]["fval"] < 0.2)
    assert np.all(calculator_results[STANDARD]["grad"] < 0.1)


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
            [1.99999999, 4.00689557, 6.23751538, 8.40233862, 11.0767572]
        ),
    }

    simulation = timepoints
    data = np.full(simulation.shape, np.nan)

    par_types = [CAT_LB, CAT_UB]

    categories = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    inner_parameter_ids = [
        f"{par_type}_{category}"
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
        OrdinalParameter(
            inner_parameter_id=inner_parameter_id,
            inner_parameter_type=InnerParameterType.ORDINAL,
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
    inner_problem = OrdinalProblem(
        xs=inner_parameters, data=[data], edatas=None, method=STANDARD
    )

    return inner_problem, expected_inner_parameter_values, simulation


def test_ordinal_solver():
    """Test the ordinal solver."""
    (
        inner_problem,
        expected_values,
        simulation,
    ) = _inner_problem_exp()

    rtol = 1e-3

    solver = OrdinalInnerSolver(
        options={METHOD: STANDARD, REPARAMETERIZED: False}
    )

    standard_result = solver.solve(
        problem=inner_problem,
        sim=[simulation],
        sigma=[np.ones(len(simulation))],
    )[0]

    assert np.allclose(
        standard_result["x"], expected_values[STANDARD], rtol=rtol
    )
    assert np.allclose(standard_result["fun"], 0, rtol=rtol)
    assert np.allclose(standard_result["jac"], 0, rtol=rtol)

    solver = OrdinalInnerSolver(
        options={METHOD: REDUCED, REPARAMETERIZED: False}
    )

    reduced_result = solver.solve(
        problem=inner_problem,
        sim=[simulation],
        sigma=[np.ones(len(simulation))],
    )[0]

    assert np.all(
        np.isclose(reduced_result["x"], expected_values[REDUCED], rtol=rtol)
    )
    assert np.allclose(reduced_result["fun"], 0, rtol=rtol)
    assert np.allclose(reduced_result["jac"], 0, rtol=rtol)


def test_surrogate_data_analytical_calculation():
    """Test analytical calculation of surrogate data.

    We test the analytical calculation of the surrogate data with respect
    to the simulation and category bounds.
    """
    inner_problem, inner_parameters, sim = _inner_problem_exp()

    rtol = 1e-3

    optimal_inner_parameters = inner_parameters[STANDARD]
    n_categories = len(optimal_inner_parameters) / 2

    expected_values = {}
    expected_values["interval_range"] = max(sim) / (2 * n_categories + 1)
    expected_values["interval_gap"] = max(sim) / (4 * (n_categories - 1) + 1)

    # As we have optimized the inner parameters, the surrogate data
    # should be the same as the simulation.
    expected_values["surrogate_data"] = sim

    options = {
        METHOD: STANDARD,
        REPARAMETERIZED: False,
        INTERVAL_CONSTRAINTS: MAX,
        MIN_GAP: 1e-16,
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
        interval_range, expected_values["interval_range"], rtol=rtol
    )
    assert np.isclose(interval_gap, expected_values["interval_gap"], rtol=rtol)
    assert np.allclose(
        surrogate_data, expected_values["surrogate_data"], rtol=rtol
    )
