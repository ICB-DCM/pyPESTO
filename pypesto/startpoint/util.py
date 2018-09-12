import numpy as np


def rescale(points, lb, ub):
    """
    Rescale points from [0, 1] to [lb, ub].

    Parameters
    ----------

    points: ndarray, shape=(n_starts, dim)
        Points in bounds [lb, ub]

    lb, ub: ndarray, shape=(1, dim)
        The boundaries, all components must be finite.
    """
    rescaled_points = points * (ub - lb) + lb
    return rescaled_points


def assign_startpoints(n_starts, startpoint_method, problem, options):
    """
    Assign startpoints.
    """
    # check if startpoints needed
    if startpoint_method is False:
        # fill with dummies
        startpoints = np.zeros(n_starts, problem.dim)
        startpoints[:] = np.nan
        return startpoints

    x_guesses = problem.x_guesses
    dim = problem.lb.size

    # number of required startpoints
    n_guessed_points = x_guesses.shape[0]
    n_required_points = n_starts - n_guessed_points

    if n_required_points <= 0:
        return x_guesses[n_starts, :]

    # apply startpoint method
    x_sampled = startpoint_method(
        n_starts=n_required_points,
        lb=problem.lb, ub=problem.ub,
        x_guesses=problem.x_guesses,
        objective=problem.objective
    )

    # put together
    startpoints = np.zeros((n_starts, dim))
    startpoints[0:n_guessed_points, :] = x_guesses
    startpoints[n_guessed_points:n_starts, :] = x_sampled

    # resample startpoints
    if options.startpoint_resample:
        startpoints = resample_startpoints(
            startpoints=startpoints,
            problem=problem,
            method=startpoint_method
        )

    return startpoints


def resample_startpoints(startpoints, problem, method):
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
            startpoint = method(
                n_starts=1,
                lb=lb,
                ub=ub,
                x_guesses=x_guesses
            )[0, :]
            fval = problem.objective(startpoint)
        # assign startpoint
        resampled_startpoints[j, :] = startpoint

    return resampled_startpoints
