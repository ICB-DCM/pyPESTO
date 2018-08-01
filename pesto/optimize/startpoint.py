import numpy as np


def uniform(n_starts, lb, ub, par_guesses=None):
    """
    Uniform sampling of start points.

    TODO: Use par_guesses.
    """
    dim = lb.shape[1]
    random_points = np.random.random((n_starts, dim))
    startpoints = random_points * (ub - lb) + lb

    return startpoints


def latin_hypercube(n_starts, lb, ub):
    """
    Latin hypercube sampling of start points.
    """
    raise NotImplementedError()
