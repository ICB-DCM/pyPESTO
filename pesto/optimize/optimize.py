import numpy as np
from pesto import Result
from .startpoint import uniform


def minimize(problem, optimizer,
             n_starts, startpoint_method=uniform, result=None) -> Result:
    """

    This is the main function to be called to perform multistart optimization.

    Parameters
    ----------

    problem: pesto.Problem
        The problem to be solved.

    optimizer: pesto.Optimizer
        The optimizer to be used n_starts times.

    n_starts: int
        Number of starts of the optimizer.

    startpoint_method: {callable, bool}
        Method for how to choose start points. False means the optimizer does
        not require start points.

    result: pesto.Result
        A result object to append the optimization results to. For example,
        one might append more runs to a previous optimization. If None,
        a new object is created.

    """

    # compute start points
    if startpoint_method is False:
        # fill with dummies
        startpoints = np.zeros(n_starts, problem.dim)
    else:
        # apply startpoint method
        startpoints = startpoint_method(n_starts,
                                        problem.lb,
                                        problem.ub,
                                        problem.par_guesses)

    # prepare result object
    if result is None:
        result = Result(problem)

    # do multistart optimization
    for j in range(0, n_starts):
        startpoint = startpoints[j, :]
        try:
            optimizer_result = optimizer.minimize(problem, startpoint)
            result.append_optimizer_result(optimizer_result=optimizer_result)
        except Exception as err:
            print(('start ' + str(j) + ' failed: {0}').format(err))

    # sort by best fval
    result.sort_optimizer_results()

    return result
