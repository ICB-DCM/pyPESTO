import numpy as np
from .util import rescale


def latin_hypercube(**kwargs):
    """
    Generate latin hypercube points.
    """

    # extract input
    n_starts = kwargs['n_starts']
    lb = kwargs['lb']
    ub = kwargs['ub']
    smooth = kwargs.get('smooth', True)

    # parse
    dim = lb.size
    lb = lb.reshape((1, -1))
    ub = ub.reshape((1, -1))

    # sample
    xs = _latin_hypercube(n_starts, dim, smooth)

    # re-scale
    xs = rescale(xs, lb, ub)

    return xs


def _latin_hypercube(n_starts, dim, smooth=True):
    """
    Generate simple latin hypercube points in [0, 1].
    """

    # uniform points
    xs = np.random.random((n_starts, dim))

    # assign sorted indices
    for j_dim in range(0, dim):
        indices = np.argsort(xs[:, j_dim])
        xs[:, j_dim] = indices

    if smooth:
        xs += np.random.random((n_starts, dim))
    else:
        xs += 0.5

    xs /= n_starts

    return xs
