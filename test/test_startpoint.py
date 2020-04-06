import numpy as np
import pypesto


# default setting
n_starts = 5
dim = 2
lb = -2 * np.ones(dim)
ub = 3 * np.ones(dim)


def test_uniform():
    xs = pypesto.startpoint.uniform(n_starts=n_starts, lb=lb, ub=ub)
    assert xs.shape == (5, 2)


def test_latin_hypercube():
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
