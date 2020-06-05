import pytest

import pypesto


@pytest.fixture
def problem():
    lb = [-5] * 10
    ub = [6] * 10
    objective = pypesto.Objective()
    problem = pypesto.Problem(
        objective=objective, lb=lb, ub=ub, x_fixed_indices=[0, 1, 5],
        x_fixed_vals=[42, 43, 44])
    return problem


def test_full_index_to_free_index(problem):
    assert problem.full_index_to_free_index(2) == 0
    assert problem.full_index_to_free_index(6) == 3
    with pytest.raises(ValueError):
        problem.full_index_to_free_index(5)
