import numpy as np

from .base import StartpointMethod
from ..problem import Problem
from ..objective import ObjectiveBase


def assign_startpoints(
    n_starts: int,
    startpoint_method: StartpointMethod,
    problem: Problem,
    startpoint_resample: bool,
) -> np.ndarray:
    """Generate start points.

    This is the main method called e.g. by `pypesto.optimize.minimize`.

    Parameters
    ----------
    n_starts:
        Number of startpoints to generate.
    startpoint_method:
        Startpoint generation method to use.
    problem:
        Underlying problem specifying e.g. dimensions and bounds.
    startpoint_resample:
        Whether to evaluate function values at proposed startpoints, and
        resample ones having non-finite values until all startpoints have
        finite value.

    Returns
    -------
    startpoints:
        Startpoints, shape (n_starts, n_par).
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
        lb=problem.lb_init,
        ub=problem.ub_init,
        objective=problem.objective,
        x_guesses=problem.x_guesses,
    )

    # put together
    startpoints = np.zeros((n_starts, dim))
    startpoints[0:n_guessed_points, :] = x_guesses
    startpoints[n_guessed_points:n_starts, :] = x_sampled

    # resample and order startpoints
    if startpoint_resample:
        startpoints = resample_startpoints(
            startpoints=startpoints,
            lb=problem.lb_init,
            ub=problem.ub_init,
            objective=problem.objective,
            x_guesses=problem.x_guesses,
            startpoint_method=startpoint_method,
        )

    return startpoints


def resample_startpoints(
    startpoints: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    objective: ObjectiveBase,
    x_guesses: np.ndarray,
    startpoint_method: StartpointMethod,
):
    """Resample startpoints having non-finite value.

    Check all proposed startpoints and resample ones with non-finite value
    via the startpoint_method.

    Also order startpoints by function value, ascending.

    Parameters
    ----------
    startpoints:
        Previously proposed startpoints.
    lb:
        Lower parameter bound.
    ub:
        Upper parameter bound.
    objective:
        Objective function, required to evaluate function values.
    x_guesses:
        Externally provided guesses, may be needed to generate remote candidate
        points.
    startpoint_method:
        Startpoint generation method to use.

    Returns
    -------
    startpoints:
        Startpoints with all finite function values, shape (n_starts, n_par).
    """

    n_starts = startpoints.shape[0]
    resampled_startpoints = np.zeros_like(startpoints)

    fvals = np.empty((n_starts,))
    # iterate over startpoints
    for j in range(0, n_starts):
        startpoint = startpoints[j, :]
        # apply method until found valid point
        objective.initialize()
        fvals[j] = objective(startpoint)
        while fvals[j] == np.inf or fvals[j] == np.nan:
            startpoint = startpoint_method(
                n_starts=1,
                lb=lb,
                ub=ub,
                objective=objective,
                x_guesses=x_guesses
            )[0, :]
            objective.initialize()
            fvals[j] = objective(startpoint)
        # assign startpoint
        resampled_startpoints[j, :] = startpoint

    # sort startpoints by function value, ascending
    startpoint_order = np.argsort(fvals)
    resampled_startpoints = resampled_startpoints[startpoint_order, :]

    return resampled_startpoints
