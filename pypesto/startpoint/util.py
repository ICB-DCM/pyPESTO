import numpy as np
from typing import Callable

from ..problem import Problem


def rescale(points, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
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


def assign_startpoints(
        n_starts: int,
        startpoint_method: Callable,
        problem: Problem,
        startpoint_resample: bool,
) -> np.ndarray:
    """
    Assign start points.
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
        return x_guesses[:n_starts, :]

    # apply startpoint method
    x_sampled = startpoint_method(
        n_starts=n_required_points,
        lb=problem.lb_init, ub=problem.ub_init,
        x_guesses=problem.x_guesses,
        objective=problem.objective
    )

    # put together
    startpoints = np.zeros((n_starts, dim))
    startpoints[0:n_guessed_points, :] = x_guesses
    startpoints[n_guessed_points:n_starts, :] = x_sampled

    # resample startpoints
    if startpoint_resample:
        startpoints = resample_startpoints(
            startpoints=startpoints,
            problem=problem,
            method=startpoint_method
        )

    return startpoints


def resample_startpoints(startpoints, problem, method):
    """
    Resample startpoints having non-finite value according to the
    startpoint_method. Also orders startpoints according to their objective
    function values (in ascending order)
    """

    n_starts = startpoints.shape[0]
    resampled_startpoints = np.zeros_like(startpoints)
    lb = problem.lb_init
    ub = problem.ub_init
    x_guesses = problem.x_guesses

    fvals = np.empty((n_starts,))
    # iterate over startpoints
    for j in range(0, n_starts):
        startpoint = startpoints[j, :]
        # apply method until found valid point
        fvals[j] = problem.objective(startpoint)
        while fvals[j] == np.inf or fvals[j] == np.nan:
            startpoint = method(
                n_starts=1,
                lb=lb,
                ub=ub,
                x_guesses=x_guesses
            )[0, :]
            fvals[j] = problem.objective(startpoint)
        # assign startpoint
        resampled_startpoints[j, :] = startpoint

    starpoint_order = np.argsort(fvals)

    return resampled_startpoints[starpoint_order, :]
