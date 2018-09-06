import numpy as np
from .util import rescale


def uniform(**kwargs):
    """
    Generate uniform points.
    """

    # extract input
    n_starts = kwargs['n_starts']
    lb = kwargs['lb']
    ub = kwargs['ub']

    # parse
    dim = lb.size
    lb = lb.reshape((1, -1))
    ub = ub.reshape((1, -1))

    # create uniform points in [0, 1]
    xs = np.random.random((n_starts, dim))

    # re-scale
    xs = rescale(xs, lb, ub)

    return xs
