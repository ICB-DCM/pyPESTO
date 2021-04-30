import numpy as np
from .util import rescale


def latin_hypercube(**kwargs) -> np.ndarray:
    """
    Generate latin hypercube points.

    Parameters
    ----------
    n_starts:
        number of starting points to be sampled.
    lb:
        lower bound.
    ub:
        upper bound.
    smooth:
        indicates if a (uniformly chosen) random starting point within the
        hypercube [i/n_starts, (i+1)/n_starts] should be chosen (True) or
        the midpoint of the interval (False). Default is True.
    """
    # extract input
    n_starts = kwargs['n_starts']
    lb = kwargs['lb']
    ub = kwargs['ub']
    smooth = kwargs.get('smooth', True)

    if not np.isfinite(ub).all() or not np.isfinite(lb).all():
        raise ValueError('Cannot use latin hypercube startpoint method with '
                         'non-finite boundaries.')

    # parse
    dim = lb.size
    lb = lb.reshape((1, -1))
    ub = ub.reshape((1, -1))

    # sample
    xs = _latin_hypercube(n_starts, dim, smooth)

    # re-scale
    xs = rescale(xs, lb, ub)

    return xs


def _latin_hypercube(
        n_starts: int,
        dim: int,
        smooth: bool = True
) -> np.ndarray:
    """
    Generate simple latin hypercube points in [0, 1].

    Parameters
    ----------

    n_starts:
        number of starting points to be sampled.

    dim:
        dimension of the optimization problem.

    smooth:
        indicates, if a (uniformly chosen) random starting point within the
        hypercube [i/n_starts, (i+1)/n_starts] should be chosen
        (`smooth==True`) or the midpoint of the interval (`smoth==False`).
    """
    xs = np.empty((n_starts, dim))

    for i_dim in range(dim):
        xs[:, i_dim] = np.random.permutation(np.arange(n_starts))

    if smooth:
        xs += np.random.random((n_starts, dim))
    else:
        xs += 0.5

    xs /= n_starts

    return xs
