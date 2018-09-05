import numpy as np


def uniform(n_starts, lb, ub, x_guesses=None):
    """
    Uniform sampling of start points.

    TODO: Use x_guesses.
    """
    dim = lb.size
    lb = lb.reshape((1, -1))
    ub = ub.reshape((1, -1))
    random_points = np.random.random((n_starts, dim))
    startpoints = random_points * (ub - lb) + lb

    return startpoints


def latin_hypercube(n_starts, lb, ub):
    """
    Latin hypercube sampling of start points.
    """
    raise NotImplementedError()


def assign_startpoints(n_starts, startpoint_method, problem, options):
    """
    Assign startpoints.
    """
    # check if startpoints needed
    if startpoint_method is False:
        # fill with dummies
        startpoints = np.zeros(n_starts, problem.dim)
        startpoints[:] = np.nan
    else:
        # apply startpoint method
        startpoints = startpoint_method(
            n_starts=n_starts,
            lb=problem.lb, ub=problem.ub,
            x_guesses=problem.x_guesses)
        # resample startpoints
        if options.startpoint_resample:
            startpoints = resample_startpoints(
                startpoints=startpoints,
                problem=problem,
                options=options)
    return startpoints


def resample_startpoints(startpoints, problem, options):
    """
    Resample startpoints having non-finite value according to the
    startpoint_method.
    """

    n_starts = startpoints.shape[0]
    resampled_startpoints = np.zeros_like(startpoints)
    lb = problem.lb
    ub = problem.ub
    x_guesses = problem.x_guesses

    # iterate over startpoints
    for j in range(0, n_starts):
        startpoint = startpoints[j, :]
        # apply method until found valid point
        fval = problem.objective(startpoint)
        while fval == np.inf or fval == np.nan:
            startpoint = options.startpoint_method(
                n_starts=1, lb=lb, ub=ub, x_guesses=x_guesses)[0, :]
            fval = problem.objective(startpoint)
        # assign startpoint
        resampled_startpoints[j, :] = startpoint

    return resampled_startpoints
