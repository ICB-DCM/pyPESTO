"""Test the startpoint methods."""

import numpy as np
import pypesto
import pytest


# default setting
n_starts = 5
dim = 2
lb = -2 * np.ones(dim)
ub = 3 * np.ones(dim)

spmethods = [pypesto.startpoint.uniform, pypesto.startpoint.latin_hypercube]


@pytest.fixture(params=spmethods)
def spmethod(request):
    return request.param


def test_uniform():
    """ Test startpoint generation using uniform sampling  """
    xs = pypesto.startpoint.uniform(n_starts=n_starts, lb=lb, ub=ub)
    assert xs.shape == (5, 2)
    assert np.all(xs >= lb)
    assert np.all(xs <= ub)


def test_latin_hypercube():
    """ Test startpoint generation using lhs sampling  """
    xs = pypesto.startpoint.latin_hypercube(
        n_starts=n_starts, lb=lb, ub=ub)
    assert xs.shape == (5, 2)

    # test latin hypercube properties
    _lb = lb.reshape((1, -1))
    _ub = ub.reshape((1, -1))
    xs = (xs - _lb) / (_ub - _lb)
    xs *= n_starts

    for j_dim in range(0, dim):
        x = xs[:, j_dim]
        x = x.astype(int)
        assert np.array_equal(sorted(x), range(0, n_starts))


def test_ubounded_startpoints(spmethod):
    """ Test Exceptions for non-finite lb/ub """
    for lb_, ub_ in [
        (-np.inf * np.ones(lb.shape), ub),
        (lb, np.inf * np.ones(ub.shape)),
        (np.nan * np.ones(lb.shape), ub),
        (lb, np.nan * np.ones(ub.shape))
    ]:
        with pytest.raises(ValueError):
            spmethod(n_starts=n_starts, lb=lb_, ub=ub_)
