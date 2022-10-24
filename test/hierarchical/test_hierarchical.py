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


def test_sth():
    petab_problem = get_problem("Boehm_JProteomeRes2014")
    # petab_problem = get_problem("Fujita_SciSignal2010")
    importer = PetabImporter(petab_problem)
    objective = importer.create_objective(hierarchical=True)
    problem = importer.create_problem(objective)
    problem.objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_adjoint
    )
    print(importer.petab_problem.x_nominal_free_scaled)
    ret = problem.objective(importer.petab_problem.x_nominal_free_scaled[:-6])
    print(ret)

    objective2 = importer.create_objective(hierarchical=False)
    problem2 = importer.create_problem(objective2)
    problem2.objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_adjoint
    )
    ret = problem2.objective(importer.petab_problem.x_nominal_free_scaled)
    print(ret)

    problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
    problem.objective.amici_solver.setRelativeTolerance(1e-8)
    problem2.objective.amici_solver.setAbsoluteTolerance(1e-8)
    problem2.objective.amici_solver.setRelativeTolerance(1e-8)
    n_starts = 50
    startpoints = pypesto.startpoint.latin_hypercube(
        n_starts=n_starts, lb=problem2.lb, ub=problem2.ub
    )
    problem.x_guesses = startpoints[:, :-6]
    print(problem.x_guesses)
    problem2.x_guesses = startpoints

    start_time = time.time()
    problem.objective.calculator.inner_solver = NumericalInnerSolver()
    engine = pypesto.MultiProcessEngine(n_procs=8)
    result = pypesto.minimize(problem, n_starts=n_starts, engine=engine)
    print(result.optimize_result.get_for_key('fval'))
    print(time.time() - start_time)

    start_time = time.time()
    problem.objective.calculator.inner_solver = AnalyticalInnerSolver()
    engine = pypesto.MultiProcessEngine(n_procs=8)
    result = pypesto.minimize(problem, n_starts=n_starts, engine=engine)
    print(result.optimize_result.get_for_key('fval'))
    print(time.time() - start_time)

    # start_time = time.time()
    # engine = pypesto.MultiProcessEngine(n_procs=8)
    # result = pypesto.minimize(problem2, n_starts=n_starts, engine=engine)
    # print(result.optimize_result.get_for_key('fval'))
    # print(time.time() - start_time)

    # TODO assert something
