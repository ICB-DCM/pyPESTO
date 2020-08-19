from ..optimize import minimize, ScipyOptimizer
from .design_problem import DesignProblem
from ..petab import PetabImporter
from ..result import Result
import numpy as np


def optimization(design_problem: DesignProblem) -> Result:
    """
    This performs a multistart optimization. As starting points use the best
    parameter from the initial optimization which are read from
    'design_problem.result.optimize_result'.

    Parameters
    ----------
    design_problem:
        the experimental design problem which contains the pypesto problem,
        the initial optimization results and the number of runs for the
        multistart optimization

    Returns
    -------
    result:
        pypesto result object

    """
    # use best optimization results as initial guesses for new optimization
    dicts = design_problem.result.optimize_result.as_list(['x'])[
            0:design_problem.n_optimize_runs]
    importer = PetabImporter(design_problem.petab_problem)

    # TODO is there a possibility to pass x_guesses while creating the problem?
    design_problem.problem = importer.create_problem(
       design_problem.problem.objective)
    design_problem.problem.x_guesses_full = np.array([d['x'] for d in dicts])

    # TODO optimizer should be chosen somewhere in the beginning
    optimizer = ScipyOptimizer()
    # engine = optimize.SingleCoreEngine()
    # engine = optimize.MultiProcessEngine()

    result = minimize(problem=design_problem.problem, optimizer=optimizer,
                      n_starts=design_problem.n_optimize_runs)

    return result
