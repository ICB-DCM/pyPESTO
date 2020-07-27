from ..optimize import minimize, ScipyOptimizer
from .design_problem import DesignProblem
from ..petab import PetabImporter
from ..result import Result


def optimization(design_problem: DesignProblem) -> Result:

    # use best optimization results as initial guesses for new optimization
    dicts = design_problem.result.optimize_result.as_list(['x'])[0:design_problem.n_optimize_runs]
    importer = PetabImporter(design_problem.petab_problem)
    design_problem.problem = importer.create_problem(design_problem.problem.objective,
                                                     x_guesses=[d['x'] for d in dicts])
    # TODO optimizer should be chosen somewhere in the beginning
    optimizer = ScipyOptimizer()
    # engine = optimize.SingleCoreEngine()
    # engine = optimize.MultiProcessEngine()

    result = minimize(problem=design_problem.problem, optimizer=optimizer,
                      n_starts=design_problem.n_optimize_runs)

    return result
