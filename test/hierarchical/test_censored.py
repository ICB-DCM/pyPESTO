from pathlib import Path

import numpy as np
import petab.v1 as petab

import pypesto
import pypesto.logging
import pypesto.optimize
import pypesto.petab
from pypesto.C import (
    INTERVAL_CENSORED,
    LEFT_CENSORED,
    LIN,
    MODE_FUN,
    RIGHT_CENSORED,
    STANDARD,
    InnerParameterType,
)
from pypesto.hierarchical.ordinal import OrdinalInnerSolver, OrdinalProblem
from pypesto.hierarchical.ordinal.parameter import OrdinalParameter

example_censored_yaml = (
    Path(__file__).parent
    / ".."
    / ".."
    / "doc"
    / "example"
    / "example_censored"
    / "example_censored.yaml"
)


def test_optimization():
    """Check that optimizations finishes without error."""
    petab_problem = petab.Problem.from_yaml(example_censored_yaml)

    optimizer = pypesto.optimize.ScipyOptimizer(
        method="L-BFGS-B", options={"maxiter": 10}
    )

    importer = pypesto.petab.PetabImporter(petab_problem, hierarchical=True)
    problem = importer.create_problem()

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


def test_ordinal_calculator_and_objective():
    """Test the ordinal calculation of objective and gradient values."""
    petab_problem = petab.Problem.from_yaml(example_censored_yaml)

    importer = pypesto.petab.PetabImporter(petab_problem, hierarchical=True)
    problem = importer.create_problem()

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

    calculator_result = calculate(problem, x_dct=x_dct)

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

    # Check that the gradient is the same as the one obtained
    # with finite differences.
    assert np.allclose(
        finite_differences_results[1],
        calculator_result["grad"],
    )


def _inner_problem_exp():
    timepoints = np.linspace(1, 10, 10)

    simulation = timepoints
    data = np.full(simulation.shape, np.nan)
    data[4:6] = [5, 6]

    par_types = ["cat_lb", "cat_ub"]

    categories = [1, 2, 4, 5, 1, 2, 4, 5]
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

    censoring_types = [
        LEFT_CENSORED,
        INTERVAL_CENSORED,
        INTERVAL_CENSORED,
        RIGHT_CENSORED,
        LEFT_CENSORED,
        INTERVAL_CENSORED,
        INTERVAL_CENSORED,
        RIGHT_CENSORED,
    ]

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
            estimate=True,
            censoring_type=censoring_type,
        )
        for inner_parameter_id, mask, inner_parameter_category, censoring_type in zip(
            inner_parameter_ids, masks, categories, censoring_types
        )
    ]

    # Add values to the inner parameters
    values = [0, 2, 6, 8, 2, 4, 8, np.inf]
    for inner_parameter, value in zip(inner_parameters, values):
        inner_parameter.value = value

    expected_values = np.asarray([0, 2, 2, 4, 6, 8, 8, np.inf])

    # Construct inner problem
    inner_problem = OrdinalProblem(
        xs=inner_parameters, data=[data], edatas=None, method=STANDARD
    )

    return inner_problem, expected_values, simulation


def test_ordinal_solver():
    """Test the ordinal solver."""
    (
        inner_problem,
        expected_values,
        simulation,
    ) = _inner_problem_exp()

    rtol = 1e-3

    solver = OrdinalInnerSolver()

    result = solver.solve(
        problem=inner_problem,
        sim=[simulation],
        sigma=[np.full(len(simulation), 1 / np.sqrt(np.pi * 2))],
    )[0]

    assert result["success"] is True
    assert np.allclose(np.asarray(result["x"]), expected_values, rtol=rtol)
    assert np.allclose(result["fun"], 0, rtol=rtol)
