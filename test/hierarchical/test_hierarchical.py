"""Tests for hierarchical optimization."""
import copy
import time

import amici
import numpy as np
import pandas as pd
import petab

import pypesto
from pypesto.C import (
    INNER_PARAMETER_BOUNDS,
    LOG10,
    LOWER_BOUND,
    MODE_FUN,
    UPPER_BOUND,
    InnerParameterType,
)
from pypesto.hierarchical.base_parameter import InnerParameter
from pypesto.hierarchical.base_problem import InnerProblem
from pypesto.hierarchical.petab import validate_hierarchical_petab_problem
from pypesto.hierarchical.relative.solver import (
    AnalyticalInnerSolver,
    NumericalInnerSolver,
)
from pypesto.hierarchical.relative.util import (
    apply_offset,
    apply_scaling,
    compute_optimal_offset,
    compute_optimal_offset_coupled,
    compute_optimal_scaling,
    compute_optimal_sigma,
)
from pypesto.optimize import FidesOptimizer, OptimizeOptions
from pypesto.petab import PetabImporter
from pypesto.testing.examples import (
    get_Boehm_JProteomeRes2014_hierarchical_petab,
    get_Boehm_JProteomeRes2014_hierarchical_petab_corrected_bounds,
)

# Suitable test cases from the benchmark collection
# - Boehm
# - Fujita


def test_hierarchical_optimization_pipeline():
    """Test hierarchical optimization of sigma and scaling parameters.

    Here (mostly): the flags `True` and `False` indicate that hierarchical
    optimization is enabled and disabled, respectively.
    """
    flags = [False, True]
    petab_problems = {
        False: get_Boehm_JProteomeRes2014_hierarchical_petab(),
        True: get_Boehm_JProteomeRes2014_hierarchical_petab_corrected_bounds(),
    }
    problems = {}
    for flag in flags:
        importer = PetabImporter(petab_problems[flag], hierarchical=flag)
        objective = importer.create_objective()
        problem = importer.create_problem(objective)
        problem.objective.amici_solver.setSensitivityMethod(
            amici.SensitivityMethod_adjoint
        )
        problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
        problem.objective.amici_solver.setRelativeTolerance(1e-8)
        problems[flag] = problem

    # Check for same optimization result
    n_starts = 1
    engine = pypesto.engine.SingleCoreEngine()
    startpoints = problems[False].get_full_vector(
        pypesto.startpoint.latin_hypercube(
            n_starts=n_starts,
            lb=problems[False].lb,
            ub=problems[False].ub,
        )
    )
    problems[False].set_x_guesses(startpoints)
    outer_indices = [
        ix
        for ix, x in enumerate(problems[False].x_names)
        if x not in problems[True].objective.calculator.get_inner_par_ids()
    ]
    problems[True].set_x_guesses(startpoints[:, outer_indices])

    inner_solvers = {
        'analytical': AnalyticalInnerSolver(),
        'numerical': NumericalInnerSolver(),
    }

    history_options = pypesto.HistoryOptions(trace_record=True)

    def get_result(problem, inner_solver_id, inner_solvers=inner_solvers):
        if inner_solver_id:
            problem.objective.calculator.inner_solver = inner_solvers[
                inner_solver_id
            ]
        start_time = time.time()
        result = pypesto.optimize.minimize(
            problem=problem,
            n_starts=n_starts,
            engine=engine,
            history_options=history_options,
            options=OptimizeOptions(allow_failed_starts=False),
            optimizer=FidesOptimizer(),
        )
        wall_time = time.time() - start_time

        best_x = result.optimize_result.list[0].x
        best_fval = result.optimize_result.list[0].fval

        result = {
            'list': result.optimize_result.list,
            'time': wall_time,
            'best_x': best_x,
            'best_fval': best_fval,
        }
        return result

    results = {}
    for problem, inner_solver_id in [
        (problems[True], 'analytical'),
        (problems[False], False),
        (problems[True], 'numerical'),
    ]:
        results[inner_solver_id] = get_result(problem, inner_solver_id)

    trace_False = np.array(
        results[False]['list'][0].history.get_fval_trace(trim=True)
    )
    trace_numerical = np.array(
        results['numerical']['list'][0].history.get_fval_trace(trim=True)
    )
    trace_analytical = np.array(
        results['numerical']['list'][0].history.get_fval_trace(trim=True)
    )

    # The analytical inner solver is at least as good as (fval / speed) the
    # numerical inner solver.
    assert at_least_as_good_as(v=trace_analytical, v0=trace_numerical)
    # The numerical inner solver is at least as good as (fval / speed) no
    # inner solver (non-hierarchical).
    assert at_least_as_good_as(v=trace_numerical, v0=trace_False)
    # Now implied that analytical is at least as good as non-hierarchical.


def test_hierarchical_calculator_and_objective():
    """Test hierarchical calculation of sigma and objective values.

    Here (mostly): the flags `True` and `False` indicate that hierarchical
    optimization is enabled and disabled, respectively.
    """
    petab_problem = (
        get_Boehm_JProteomeRes2014_hierarchical_petab_corrected_bounds()
    )
    flags = [False, True]
    problems = {}
    for flag in flags:
        importer = PetabImporter(petab_problem, hierarchical=flag)
        objective = importer.create_objective()
        problem = importer.create_problem(objective)
        problem.objective.amici_solver.setSensitivityMethod(
            amici.SensitivityMethod_adjoint
        )
        problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
        problem.objective.amici_solver.setRelativeTolerance(1e-8)
        problems[flag] = problem

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
    # Nominal sigma values are close to optimal.
    # One is changed here to facilitate testing.
    x_dct['sd_pSTAT5A_rel'] = 0.5

    calculator_results = {
        flag: calculate(problems[flag], x_dct=x_dct) for flag in flags
    }

    # Hierarchical optimization means that the results differ here, because
    # the `False` case has suboptimal sigma values.
    assert not np.isclose(
        calculator_results[True]['fval'],
        calculator_results[False]['fval'],
    )
    assert not np.isclose(
        calculator_results[True]['grad'],
        calculator_results[False]['grad'],
    ).all()

    for inner_idx, inner_par in enumerate(
        problems[True].objective.calculator.get_inner_par_ids()
    ):
        x_dct[inner_par] = calculator_results[True]['inner_parameters'][
            inner_idx
        ]
    calculator_results[False] = calculate(problem=problems[False], x_dct=x_dct)

    # The `False` case has copied the optimal sigma values from hierarchical
    # optimization, so can produce the same results now.
    assert np.isclose(
        calculator_results[True]['fval'],
        calculator_results[False]['fval'],
    )
    assert np.isclose(
        calculator_results[True]['grad'],
        calculator_results[False]['grad'],
    ).all()

    parameters = [x_dct[x_id] for x_id in petab_problem.x_free_ids]
    fval_false = problems[False].objective(parameters)

    outer_parameters = [
        x_dct[x_id] for x_id in problems[True].objective.x_names
    ]
    fval_true = problems[True].objective(outer_parameters)
    # Hierarchical optimization does not affect the function value, if optimal
    # sigma are provided to the normal function. High precision is required as
    # the nominal values are very good already, so the test might pass
    # accidentally if the nominal values are used accidentally.
    assert np.isclose(fval_true, fval_false, atol=1e-12, rtol=1e-14)


def test_analytical_computations():
    """Test analytically-solved hierarchical inner parameters."""
    function = np.exp

    timepoints = np.linspace(0, 10, 101)

    simulation = function(timepoints)
    dummy_sigma = np.ones(simulation.shape)
    mask = np.full(simulation.shape, True)

    expected_scaling_value = 5
    expected_offset_value = 2
    expected_sigma_value = 2

    rtol = 1e-3

    # Scaling
    simulation = function(timepoints)
    data = expected_scaling_value * simulation
    scaling_value = compute_optimal_scaling(
        data=[data],
        sim=[simulation],
        sigma=[dummy_sigma],
        mask=[mask],
    )
    assert np.isclose(scaling_value, expected_scaling_value, rtol=rtol)

    # Offset
    simulation = function(timepoints)
    data = simulation + expected_offset_value
    offset_value = compute_optimal_offset(
        data=[data],
        sim=[simulation],
        sigma=[dummy_sigma],
        mask=[mask],
    )
    assert np.isclose(offset_value, expected_offset_value, rtol=rtol)

    # Coupled (scaling and offset)
    simulation = function(timepoints)
    data = expected_scaling_value * simulation + expected_offset_value
    offset_value = compute_optimal_offset_coupled(
        data=[data],
        sim=[simulation],
        sigma=[dummy_sigma],
        mask=[mask],
    )
    apply_offset(offset_value=expected_offset_value, data=[data], mask=[mask])
    scaling_value = compute_optimal_scaling(
        data=[data],
        sim=[simulation],
        sigma=[dummy_sigma],
        mask=[mask],
    )
    assert np.isclose(offset_value, expected_offset_value, rtol=rtol)
    assert np.isclose(scaling_value, expected_scaling_value, rtol=rtol)

    # All (scaling, offset, sigma)
    simulation = function(timepoints)

    data = expected_scaling_value * simulation + expected_offset_value
    data[0::2] -= expected_sigma_value
    data[1::2] += expected_sigma_value

    offset_value = compute_optimal_offset_coupled(
        data=[data],
        sim=[simulation],
        sigma=[dummy_sigma],
        mask=[mask],
    )
    apply_offset(offset_value=offset_value, data=[data], mask=[mask])
    scaling_value = compute_optimal_scaling(
        data=[data],
        sim=[simulation],
        sigma=[dummy_sigma],
        mask=[mask],
    )
    apply_scaling(scaling_value=scaling_value, sim=[simulation], mask=[mask])
    sigma_value = compute_optimal_sigma(data=data, sim=simulation, mask=mask)

    assert np.isclose(offset_value, expected_offset_value, rtol=rtol)
    assert np.isclose(scaling_value, expected_scaling_value, rtol=rtol)
    assert np.isclose(sigma_value, expected_sigma_value, rtol=rtol)


def inner_problem_exp(add_scaling: bool = True, add_offset: bool = True):
    function = np.exp
    timepoints = np.linspace(0, 3, 101)

    expected_values = {
        'scaling_': 5,
        'offset_': 2,
        'sigma_': 3,
    }

    simulation = function(timepoints)

    data = copy.deepcopy(simulation)
    if add_scaling:
        data *= expected_values['scaling_']

    if add_offset:
        data += expected_values['offset_']

    data[0::2] -= expected_values['sigma_']
    data[1::2] += expected_values['sigma_']

    mask = np.full(data.shape, True)

    inner_parameters = [
        InnerParameter(
            inner_parameter_id=inner_parameter_id,
            inner_parameter_type=inner_parameter_type,
            scale=LOG10,
            lb=INNER_PARAMETER_BOUNDS[inner_parameter_type][LOWER_BOUND],
            ub=INNER_PARAMETER_BOUNDS[inner_parameter_type][UPPER_BOUND],
            ixs=mask,
        )
        for inner_parameter_id, inner_parameter_type in [
            ('offset_', InnerParameterType.OFFSET)
            if add_offset
            else (None, None),
            ('scaling_', InnerParameterType.SCALING)
            if add_scaling
            else (None, None),
            ('sigma_', InnerParameterType.SIGMA),
        ]
        if inner_parameter_id is not None
    ]

    if add_scaling and add_offset:
        inner_parameters[0].coupled = inner_parameters[1]
        inner_parameters[1].coupled = inner_parameters[0]

    inner_problem = InnerProblem(xs=inner_parameters, data=[data])

    return inner_problem, expected_values, simulation


def test_analytical_inner_solver():
    """Test numerically-solved hierarchical inner parameters."""
    inner_problem, expected_values, simulation = inner_problem_exp()

    dummy_sigma = np.ones(simulation.shape)

    rtol = 1e-3

    solver = AnalyticalInnerSolver()

    result = solver.solve(
        problem=inner_problem,
        sim=[simulation],
        sigma=[dummy_sigma],
        scaled=False,
    )

    assert np.isclose(result['offset_'], expected_values['offset_'], rtol=rtol)
    assert np.isclose(
        result['scaling_'], expected_values['scaling_'], rtol=rtol
    )
    assert np.isclose(result['sigma_'], expected_values['sigma_'], rtol=rtol)


def test_numerical_inner_solver():
    """Test numerically-solved hierarchical inner parameters."""
    inner_problem, expected_values, simulation = inner_problem_exp()

    dummy_sigma = np.ones(simulation.shape)

    rtol = 1e-3

    solver = NumericalInnerSolver(minimize_kwargs={'n_starts': 10})
    result = solver.solve(
        problem=inner_problem,
        sim=[simulation],
        sigma=[dummy_sigma],
        scaled=False,
    )

    assert np.isclose(result['offset_'], expected_values['offset_'], rtol=rtol)
    assert np.isclose(
        result['scaling_'], expected_values['scaling_'], rtol=rtol
    )
    assert np.isclose(result['sigma_'], expected_values['sigma_'], rtol=rtol)


def test_non_coupled_analytical_inner_solver():
    """Test analytically-solved non-coupled hierarchical inner parameters."""
    # Test for only offset
    inner_problem, expected_values, simulation = inner_problem_exp(
        add_scaling=False
    )
    dummy_sigma = np.ones(simulation.shape)

    rtol = 1e-1

    solver = AnalyticalInnerSolver()
    result = solver.solve(
        problem=inner_problem,
        sim=[simulation],
        sigma=[dummy_sigma],
        scaled=False,
    )
    assert np.isclose(result['offset_'], expected_values['offset_'], rtol=rtol)
    assert np.isclose(result['sigma_'], expected_values['sigma_'], rtol=rtol)

    # Test for only scaling
    inner_problem, expected_values, simulation = inner_problem_exp(
        add_offset=False
    )
    dummy_sigma = np.ones(simulation.shape)

    rtol = 1e-3

    solver = AnalyticalInnerSolver()
    result = solver.solve(
        problem=inner_problem,
        sim=[simulation],
        sigma=[dummy_sigma],
        scaled=False,
    )

    assert np.isclose(
        result['scaling_'], expected_values['scaling_'], rtol=rtol
    )
    assert np.isclose(result['sigma_'], expected_values['sigma_'], rtol=rtol)


def test_constrained_inner_solver():
    """Test numerically- and analytically-solved box-constrained hierarchical inner parameters."""
    inner_problem, expected_values, simulation = inner_problem_exp()

    dummy_sigma = np.ones(simulation.shape)

    all_lb = [(6, 3), (3, 0), (3, 1), (4, 3)]
    all_ub = [(7, 4), (4, 1), (4, 3), (6, 4)]

    all_expected_values = [
        {'scaling_': 6, 'offset_': 3},
        {'scaling_': 4, 'offset_': 1},
        {
            'scaling_': 4,  # all_lb[2][0],
            'offset_': np.clip(
                compute_optimal_offset(
                    data=inner_problem.data,
                    sim=[simulation],
                    sigma=[dummy_sigma],
                    mask=[np.full(simulation.shape, True)],
                    optimal_scaling=4.0,
                ),
                1,  # all_lb[2][1],
                3,  # all_ub[2][1],
            ),
        },
        {
            'scaling_': np.clip(
                compute_optimal_scaling(
                    data=inner_problem.data,
                    sim=[simulation],
                    sigma=[dummy_sigma],
                    mask=[np.full(simulation.shape, True)],
                    optimal_offset=3.0,
                ),
                4,  # all_lb[3][0],
                6,  # all_ub[3][0],
            ),
            'offset_': 3,  # all_lb[3][1],
        },
    ]

    for lb, ub, expected_values in zip(all_lb, all_ub, all_expected_values):
        # Set seed for reproducibility
        np.random.seed(1)
        inner_problem.get_for_id('scaling_').lb = lb[0]
        inner_problem.get_for_id('scaling_').ub = ub[0]
        inner_problem.get_for_id('offset_').lb = lb[1]
        inner_problem.get_for_id('offset_').ub = ub[1]

        copied_sim = copy.deepcopy(simulation)
        rtol = 1e-3

        solver = AnalyticalInnerSolver()
        ana_res = solver.solve(
            problem=inner_problem,
            sim=[copied_sim],
            sigma=[dummy_sigma],
            scaled=False,
        )

        copied_sim = copy.deepcopy(simulation)
        solver = NumericalInnerSolver(minimize_kwargs={'n_starts': 10})
        num_res = solver.solve(
            problem=inner_problem,
            sim=[copied_sim],
            sigma=[dummy_sigma],
            scaled=False,
        )

        assert np.isclose(ana_res['offset_'], num_res['offset_'], rtol=rtol)
        assert np.isclose(ana_res['scaling_'], num_res['scaling_'], rtol=rtol)

        assert np.isclose(
            ana_res['offset_'], expected_values['offset_'], rtol=rtol
        )
        assert np.isclose(
            ana_res['scaling_'], expected_values['scaling_'], rtol=rtol
        )


def test_non_coupled_constrained_inner_solver():
    """Test non-coupled box-constrained hierarchical inner parameters."""
    for current_par, add_scaling, add_offset, lb, ub in zip(
        ['scaling_', 'scaling_', 'offset_', 'offset_'],
        [True, True, False, False],
        [False, False, True, True],
        [6, None, 3, None],
        [None, 4, None, 1],
    ):
        # Set seed for reproducibility
        np.random.seed(4)
        inner_problem, expected_values, simulation = inner_problem_exp(
            add_scaling=add_scaling,
            add_offset=add_offset,
        )
        if lb is not None:
            inner_problem.get_for_id(current_par).lb = lb
            expected_values = {current_par: lb}
        if ub is not None:
            inner_problem.get_for_id(current_par).ub = ub
            expected_values = {current_par: ub}

        dummy_sigma = np.ones(simulation.shape)
        copied_sim = copy.deepcopy(simulation)
        rtol = 1e-3

        solver = AnalyticalInnerSolver()
        ana_res = solver.solve(
            problem=inner_problem,
            sim=[copied_sim],
            sigma=[dummy_sigma],
            scaled=False,
        )

        copied_sim = copy.deepcopy(simulation)
        solver = NumericalInnerSolver(minimize_kwargs={'n_starts': 10})
        num_res = solver.solve(
            problem=inner_problem,
            sim=[copied_sim],
            sigma=[dummy_sigma],
            scaled=False,
        )

        assert np.isclose(
            ana_res[current_par], num_res[current_par], rtol=rtol
        )

        assert np.isclose(
            ana_res[current_par], expected_values[current_par], rtol=rtol
        )


def at_least_as_good_as(v, v0) -> bool:
    """Check that the first vector of fvals is at least as good the second.

    Parameters
    ----------
    v:
        The first vector of fvals.
    v0:
        The second vector of fvals.

    Returns
    -------
    Whether the first vector of fvals is at least as good as the second.
    """
    max_index = min(len(v), len(v0))
    return (v[:max_index] <= v0[:max_index]).all()


def test_validate():
    # Scaling shared across multiple observables - okay
    observable_df = petab.get_observable_df(
        pd.DataFrame(
            {
                petab.OBSERVABLE_ID: ["obs1", "obs2"],
                petab.OBSERVABLE_FORMULA: [
                    "observableParameter1_obs1 * x1",
                    "observableParameter1_obs2 * x2",
                ],
                petab.NOISE_FORMULA: [
                    "noiseParameter1_obs1",
                    "noiseParameter1_obs2",
                ],
            }
        )
    )
    measurement_df = petab.get_measurement_df(
        pd.DataFrame(
            {
                petab.OBSERVABLE_ID: ["obs1", "obs2"],
                petab.TIME: [0, 1],
                petab.MEASUREMENT: [1, 2],
                petab.OBSERVABLE_PARAMETERS: ["s", "s"],
                petab.NOISE_PARAMETERS: [0.1, 0.1],
            }
        )
    )
    parameter_df = petab.get_parameter_df(
        pd.DataFrame(
            {
                petab.PARAMETER_ID: ["s"],
                "parameterType": ['scaling'],
                "estimate": [1],
            }
        )
    )
    petab_problem = petab.Problem(
        observable_df=observable_df,
        parameter_df=parameter_df,
        measurement_df=measurement_df,
    )
    validate_hierarchical_petab_problem(petab_problem)
