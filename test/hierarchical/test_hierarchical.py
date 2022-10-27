import time

import amici
import fides
import numpy as np
import pandas as pd
import petab
from benchmark_models_petab import get_problem

import pypesto

# import pypesto.logging
from pypesto.C import MODE_FUN
from pypesto.hierarchical.parameter import InnerParameter
from pypesto.hierarchical.problem import PARAMETER_TYPE
from pypesto.hierarchical.solver import (
    AnalyticalInnerSolver,
    NumericalInnerSolver,
)
from pypesto.petab import PetabImporter

# pypesto.logging.log_to_console(level=logging.DEBUG)

# TODO
# - test scaling and offset parameters

# Suitable test cases from the benchmark collection
# - Boehm
# - Fujita


def get_boehm():
    petab_problem = get_problem("Boehm_JProteomeRes2014")
    # Add scaling and offset parameters
    petab_problem.observable_df[petab.OBSERVABLE_FORMULA] = [
        f"observableParameter2_{obs_id} + observableParameter1_{obs_id} "
        f"* {obs_formula}"
        for obs_id, obs_formula in zip(
            petab_problem.observable_df.index,
            petab_problem.observable_df[petab.OBSERVABLE_FORMULA],
        )
    ]
    # Set scaling and offset parameters for measurements
    assert (
        petab_problem.measurement_df[petab.OBSERVABLE_PARAMETERS].isna().all()
    )
    petab_problem.measurement_df[petab.OBSERVABLE_PARAMETERS] = [
        f"scaling_{obs_id};offset_{obs_id}"
        for obs_id in petab_problem.measurement_df[petab.OBSERVABLE_ID]
    ]
    # Add output parameters to parameter table
    extra_parameters = []
    for par_id in (
        'offset_pSTAT5A_rel',
        'offset_pSTAT5B_rel',
        'offset_rSTAT5A_rel',
    ):
        extra_parameters.append(
            {
                petab.PARAMETER_ID: par_id,
                petab.PARAMETER_SCALE: petab.LIN,
                petab.LOWER_BOUND: -100,
                petab.UPPER_BOUND: 100,
                petab.NOMINAL_VALUE: 0,
                petab.ESTIMATE: 0,
            }
        )
    for par_id, nominal_value in zip(
        ('scaling_pSTAT5A_rel', 'scaling_pSTAT5B_rel', 'scaling_rSTAT5A_rel'),
        (3.85261197844677, 6.59147818673419, 3.15271275648527),
    ):
        extra_parameters.append(
            {
                petab.PARAMETER_ID: par_id,
                petab.PARAMETER_SCALE: petab.LOG10,
                petab.LOWER_BOUND: 1e-5,
                petab.UPPER_BOUND: 1e5,
                petab.NOMINAL_VALUE: nominal_value,
                petab.ESTIMATE: 1,
            }
        )

    petab_problem.parameter_df = pd.concat(
        [
            petab_problem.parameter_df,
            petab.get_parameter_df(pd.DataFrame(extra_parameters)),
        ]
    )
    # Mark output parameters for hierarchical optimization
    petab_problem.parameter_df[PARAMETER_TYPE] = None
    for par_id in petab_problem.parameter_df.index:
        if par_id.startswith("offset_"):
            petab_problem.parameter_df.loc[
                par_id, PARAMETER_TYPE
            ] = InnerParameter.OFFSET
        elif par_id.startswith("sd_"):
            petab_problem.parameter_df.loc[
                par_id, PARAMETER_TYPE
            ] = InnerParameter.SIGMA
        elif par_id.startswith("scaling_"):
            petab_problem.parameter_df.loc[
                par_id, PARAMETER_TYPE
            ] = InnerParameter.SCALING
    petab.lint_problem(petab_problem)

    return petab_problem


def test_hierarchical_optimization_sigma_and_scaling():
    """Test hierarchical optimization of sigma and scaling parameters.

    Here (mostly): the flags `True` and `False` indicate that hierarchical
    optimization is enabled and disabled, respectively.
    """
    petab_problem = get_boehm()
    importer = PetabImporter(petab_problem)
    flags = [False, True]
    problems = {}
    for flag in flags:
        objective = importer.create_objective(hierarchical=flag)
        problem = importer.create_problem(objective)
        problem.objective.amici_solver.setSensitivityMethod(
            amici.SensitivityMethod_adjoint
        )
        problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
        problem.objective.amici_solver.setRelativeTolerance(1e-8)
        problems[flag] = problem

    # Check for same optimization result
    n_starts = 24
    engine = pypesto.engine.MultiThreadEngine()
    startpoints = problems[False].get_full_vector(
        pypesto.startpoint.latin_hypercube(
            n_starts=n_starts,
            lb=problems[False].lb,
            ub=problems[False].ub,
        )
    )

    problems[False].set_x_guesses(startpoints)
    # FIXME ideally would not need to provide guesses for inner parameters
    problems[True].set_x_guesses(startpoints)

    inner_solvers = {
        'analytical': AnalyticalInnerSolver(),
        'numerical': NumericalInnerSolver(),
    }

    history_options = pypesto.HistoryOptions(trace_record=True)
    optimizer = pypesto.optimize.FidesOptimizer(
        verbose=0, hessian_update=fides.BFGS()
    )

    def get_result(problem, inner_solver, inner_solvers=inner_solvers):
        if inner_solver:
            problem.objective.calculator.inner_solver = inner_solvers[
                inner_solver_id
            ]

        start_time = time.time()
        result = pypesto.optimize.minimize(
            problem=problem,
            n_starts=n_starts,
            engine=engine,
            history_options=history_options,
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
        (problems[False], False),
        (problems[True], 'analytical'),
        (problems[True], 'numerical'),
    ]:
        results[inner_solver_id] = get_result(problem, inner_solver_id)

    # DEBUGGING
    f = results[False]['list']
    a = results['analytical']['list']
    n = results['numerical']['list']
    f_s = sorted(f, key=lambda s: int(s.id))
    a_s = sorted(a, key=lambda s: int(s.id))
    n_s = sorted(n, key=lambda s: int(s.id))
    f_v = [s.fval for s in f_s]
    a_v = [s.fval for s in a_s]
    n_v = [s.fval for s in n_s]
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax.plot(range(n_starts), f_v, label='not hierarchical')
    ax.plot(range(n_starts), a_v, label='analytical hierarchical')
    ax.plot(range(n_starts), n_v, label='numerical hierarchical')
    ax.legend()
    plt.show()

    for i in range(n_starts):
        _, ax = plt.subplots()

        ax.plot((results[False]['list'][i].history.get_fval_trace(trim=True)), label='not hierarchical')
        ax.plot((results['analytical']['list'][i].history.get_fval_trace(trim=True)), label='analytical hierarchical')
        ax.plot((results['numerical']['list'][i].history.get_fval_trace(trim=True)), label='numerical hierarchical')
        ax.set_yscale("log")
        ax.legend()
        plt.show()


    breakpoint()
    # END DEBUGGING

    assert results['analytical']['time'] < results[False]['time']
    assert results['numerical']['time'] < results[False]['time']
    assert results['analytical']['time'] < results['numerical']['time']

    assert np.isclose(
        results['analytical']['best_fval'], results[False]['best_fval']
    )
    assert np.isclose(
        results['analytical']['best_fval'], results['numerical']['best_fval']
    )
    # Then `numerical isclose False` is mostly implied.

    # TODO assert optimized vector is similar at both model and objective
    #      hyperparameter values.


def test_hierarchical_calculator_and_objective():
    """Test hierarchical calculation of sigma and objective values.

    Here (mostly): the flags `True` and `False` indicate that hierarchical
    optimization is enabled and disabled, respectively.
    """
    petab_problem = get_boehm()
    importer = PetabImporter(petab_problem)
    flags = [False, True]
    problems = {}
    for flag in flags:
        objective = importer.create_objective(hierarchical=flag)
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
    # Nominal sigma values are close to optimal. One is changed here to facilitate testing.
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

    x_dct.update(calculator_results[True]['inner_parameters'])
    calculator_results[False] = calculate(problem=problems[False], x_dct=x_dct)

    # The `False` case has copied the optimal sigma values from hierarchical optimization,
    # so can produce the same results now.
    assert np.isclose(
        calculator_results[True]['fval'],
        calculator_results[False]['fval'],
    )
    assert np.isclose(
        calculator_results[True]['grad'],
        calculator_results[False]['grad'],
    ).all()

    parameters = [x_dct[x_id] for x_id in petab_problem.x_free_ids]
    fval_False = problems[False].objective(parameters)

    # TODO user-friendly way to get these
    outer_parameters = [
        x_dct[x_id]
        for x_id in petab_problem.x_free_ids
        if pd.isna(petab_problem.parameter_df.loc[x_id].parameterType)
    ]
    fval_True = problems[True].objective(outer_parameters)
    # Hierarchical optimization does not affect the function value, if optimal sigma are provided to the normal function.
    # High precision is required as the nominal values are very good already, so the test might pass accidentally
    # if the nominal values are used accidentally.
    assert np.isclose(fval_True, fval_False, atol=1e-12, rtol=1e-14)
