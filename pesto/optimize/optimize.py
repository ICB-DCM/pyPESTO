import numpy as np
from pesto import Result

def minimize(problem, optimizer,
             n_starts, startpoint_method=False, result=None) -> Result:
    """
    Parameters
    ----------

    problem: pesto.Problem

    optimizer: pesto.Optimizer

    n_starts: int
        Number of starts

    startpoint_method: {callable, bool}
        Method for how to choose start points. False means the optimizer does
        not require start points.

    result: pesto.Result
        A result object to append the optiization results to. If None,
        a new object is created.

    """

    # compute start points
    if startpoint_method is not False:
        startpoints = startpoint_method(n_starts,
                                        problem.lb,
                                        problem.ub,
                                        problem.par_guesses)
    else:
        startpoints = np.zeros(n_starts, 1)

    # prepare result object
    if result is None:
        result = Result()
    result.problem = problem

    optimizer_results = []

    # do multistart optimization
    for j in range(0, n_starts):
        startpoint = startpoints[j]
        optimizer_result = optimizer.minimize(problem, startpoint)
        optimizer_results.append(optimizer_result)

    result.optimization = optimizer_results

    return result
