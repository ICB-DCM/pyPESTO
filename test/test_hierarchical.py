import pypesto
import pypesto.logging
import logging
import amici
import time

from pypesto.hierarchical.solver import AnalyticalInnerSolver

#pypesto.logging.log_to_console(level=logging.DEBUG)


def test_sth():
    yaml_file = "/home/yannik/benchmark-models-petab/Benchmark-Models/Boehm_JProteomeRes2014/Boehm_JProteomeRes2014.yaml"
    #yaml_file = "/home/yannik/benchmark-models-petab/Benchmark-Models/Fujita_SciSignal2010/Fujita_SciSignal2010.yaml"
    importer = pypesto.PetabImporter.from_yaml(yaml_file)
    objective = importer.create_objective(hierarchical=True)
    problem = importer.create_problem(objective)
    problem.objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
    print(importer.petab_problem.x_nominal_free_scaled)
    ret = problem.objective(importer.petab_problem.x_nominal_free_scaled[:-3])
    print(ret)

    objective2 = importer.create_objective(hierarchical=False)
    problem2 = importer.create_problem(objective2)
    problem2.objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
    ret = problem2.objective(importer.petab_problem.x_nominal_free_scaled)
    print(ret)

    problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
    problem.objective.amici_solver.setRelativeTolerance(1e-8)
    problem2.objective.amici_solver.setAbsoluteTolerance(1e-8)
    problem2.objective.amici_solver.setRelativeTolerance(1e-8)

    startpoints = pypesto.startpoint.latin_hypercube(n_starts=40, lb=problem2.lb, ub=problem2.ub)
    problem.x_guesses = startpoints[:, :-3]
    print(problem.x_guesses)
    problem2.x_guesses = startpoints

    start_time = time.time()
    engine = pypesto.MultiProcessEngine(n_procs=8)
    result = pypesto.minimize(problem, n_starts=40, engine=engine)
    print(result.optimize_result.get_for_key('fval'))
    print(time.time() - start_time)

    start_time = time.time()
    problem.objective.calculator.inner_solver = AnalyticalInnerSolver()
    engine = pypesto.MultiProcessEngine(n_procs=8)
    result = pypesto.minimize(problem, n_starts=40, engine=engine)
    print(result.optimize_result.get_for_key('fval'))
    print(time.time() - start_time)

    start_time = time.time()
    engine = pypesto.MultiProcessEngine(n_procs=8)
    result = pypesto.minimize(problem2, n_starts=40, engine=engine)
    print(result.optimize_result.get_for_key('fval'))
    print(time.time() - start_time)