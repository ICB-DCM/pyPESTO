import logging
import time

import amici
from benchmark_models_petab import get_problem

import pypesto
import pypesto.logging
from pypesto.hierarchical.solver import (
    AnalyticalInnerSolver,
    NumericalInnerSolver,
)
from pypesto.petab import PetabImporter

# pypesto.logging.log_to_console(level=logging.DEBUG)


# TODO
# - test scaling and offset parameters


def test_hierarchical_sigma():
    """Test hierarchical optimization of noise (sigma) parameters.

    Here (mostly): the flags `True` and `False` indicate that hierarchical
    optimization is enabled and disabled, respectively.
    """

    petab_problem = get_problem("Boehm_JProteomeRes2014")
    # petab_problem = get_problem("Fujita_SciSignal2010")

    importer = PetabImporter(petab_problem)

    # `True` indicates hierarchical optimization is enabled,
    # `False` is without.
    flags = [False, True]
    startpoints = None

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

    # Check for same fval
    # TODO get sigma from hierarchical optimization then supply to non-hierarchical,
    #      for fair test.
    fval_False = problem[False].objective(importer.petab_problem.x_nominal_free_scaled)
    fval_True = problem[True].objective(importer.petab_problem.x_nominal_free_scaled[:6])
    # Hierarchical optimization does not affect the function value.
    assert fval_True == fval_False


    # Check for same optimization result
    startpoints = pypesto.startpoint.latin_hypercube(
        n_starts=n_starts, lb=problems[False].lb, ub=problems[False].ub,
    )

    problems[False].set_x_guesses(startpoints)
    problems[True].set_x_guesses(startpoints[:, :6])

    inner_solvers = {
        'analytical': AnalyticalInnerSolver(),
        'numerical': NumericalInnerSolver(),
    ]

    def get_result(problem, inner_solver, inner_solvers=inner_solvers):
        if inner_solver:
            problem.objective.calculator.inner_solver = inner_solvers[inner_solver_id]

        engine = pypesto.MultiProcessEngine(n_procs=8)

        start_time = time.time()
        result = pypesto.minimize(problem, n_starts=50, engine=engine)
        time = time.time() - start_time
        
        best_x = result.optimize_result.list[0].x
        best_fval = result.optimize_result.list[0].fval

        result = {
            'time': time,
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

    assert results['analytical']['time'] < results[False]['time']
    assert results['numerical']['time'] < results[False]['time']
    assert results['analytical']['time'] < results['numerical']['time']

    assert np.isclose(results['analytical']['best_fval'], results[False]['best_fval'])
    assert np.isclose(results['analytical']['best_fval'], results['numerical']['best_fval'])
    # Then `numerical isclose False` is mostly implied.

    # TODO assert optimized vector is similar at both model and objective
    #      hyperparameter values.
