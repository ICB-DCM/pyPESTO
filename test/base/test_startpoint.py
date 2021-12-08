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
        n_starts=n_starts, lb=lb, ub=ub
    )
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


def test_unbounded_startpoints(spmethod):
    """Test Exceptions for non-finite lb/ub"""
    for lb_, ub_ in [
        (-np.inf * np.ones(lb.shape), ub),
        (lb, np.inf * np.ones(ub.shape)),
        (np.nan * np.ones(lb.shape), ub),
        (lb, np.nan * np.ones(ub.shape))
    ]:
        with pytest.raises(ValueError):
            spmethod(n_starts=n_starts, lb=lb_, ub=ub_)


@pytest.mark.parametrize("check_fval", [True, False])
@pytest.mark.parametrize("check_grad", [True, False])
def test_resampling(check_fval: bool, check_grad: bool):
    """Test that startpoint resampling works."""
    dim = 3
    lb = -1 * np.ones(shape=dim)
    ub = 1 * np.ones(shape=dim)

    def fun(x: np.ndarray):
        if x[0] > 0.5:
            return np.nan
        if x[1] > 0.5:
            return np.inf
        if x[2] > 0.5:
            return - np.inf
        return np.sum(x)

    def grad(x: np.ndarray):
        if x[0] < - 0.5:
            return np.full_like(x, fill_value=np.nan)
        if x[1] < - 0.5:
            return np.full_like(x, fill_value=np.inf)
        if x[2] < - 0.5:
            return np.full_like(x, fill_value=- np.inf)
        return x

    # startpoint guesses
    n_guesses = 20
    x_guesses = pypesto.startpoint.uniform(n_starts=n_guesses, lb=lb, ub=ub)

    # define objective and problem
    obj = pypesto.Objective(fun=fun, grad=grad)
    problem = pypesto.Problem(objective=obj, lb=lb, ub=ub, x_guesses=x_guesses)

    # define startpoint method (here only considering uniform for simplicity)
    startpoint_method = pypesto.startpoint.UniformStartpoints(
        use_guesses=True,
        check_fval=check_fval,
        check_grad=check_grad,
    )

    # find startpoints
    xs = startpoint_method(n_starts=40, problem=problem)

    # calculate function values and gradients
    fvals = np.array([fun(x) for x in xs])
    grads = np.array([grad(x) for x in xs])

    # check that function values are (not) finite
    if check_fval:
        assert np.isfinite(fvals).all()
    else:
        assert not np.isfinite(fvals).all()

    # check that gradients are (not) finite
    if check_grad:
        assert np.isfinite(grads).all()
    else:
        assert not np.isfinite(grads).all()

    # check that guesses were used and potentially discarded
    if check_fval or check_grad:
        assert not np.allclose(x_guesses, xs[:n_guesses, :])
    else:
        assert np.allclose(x_guesses, xs[:n_guesses, :])
