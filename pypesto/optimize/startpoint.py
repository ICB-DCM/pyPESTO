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
