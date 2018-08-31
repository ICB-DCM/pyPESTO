import numpy as np
from pypesto import Result
from .startpoint import uniform


def minimize(problem, optimizer,
             n_starts,
             startpoint_method=uniform,
             result=None,
             startpoint_resampling=True,
             allow_failed_starts=True) -> Result:
    """

    This is the main function to be called to perform multistart optimization.

    Parameters
    ----------

    problem: pypesto.Problem
        The problem to be solved.

    optimizer: pypesto.Optimizer
        The optimizer to be used n_starts times.

    n_starts: int
        Number of starts of the optimizer.

    startpoint_method: {callable, bool}
        Method for how to choose start points. False means the optimizer does
        not require start points.

    startpoint_resampling: bool
        Flag indicating whether initial points are supposed to be resampled if
        function evaluation fails at the initial point

    result: pypesto.Result
        A result object to append the optimization results to. For example,
        one might append more runs to a previous optimization. If None,
        a new object is created.

    allow_failed_starts: bool
        Flag indicating whether we tolerate that exceptions are thrown during
        the minimization process.


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
        if startpoint_resampling:
            valid_startpoint = problem.objective(startpoint) < float('inf')
            while not valid_startpoint:
                startpoint = startpoint_method(
                    1,
                    problem.lb,
                    problem.ub,
                    problem.par_guesses
                )[0, :]
                valid_startpoint = problem.objective(startpoint) < float('inf')

        try:
            optimizer_result = optimizer.minimize(problem, startpoint, j)
        except Exception as err:
            if allow_failed_starts:
                print(('start ' + str(j) + ' failed: {0}').format(err))
                optimizer_result = optimizer.recover_result(
                    problem,
                    startpoint,
                    err
                )
            else:
                raise

        result.optimize_result.append(optimizer_result=optimizer_result)

    # sort by best fval
    result.optimize_result.sort()

    return result
