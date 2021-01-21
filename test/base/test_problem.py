import pytest

import pypesto


@pytest.fixture
def problem():
    """A very basic problem."""
    lb = [-5] * 10
    ub = [6] * 10
    objective = pypesto.Objective()
    problem = pypesto.Problem(
        objective=objective, lb=lb, ub=ub, x_fixed_indices=[0, 1, 5],
        x_fixed_vals=[42, 43, 44])
    return problem


def test_dim_vs_dim_full():
    """Test passing arrays in the full or reduced dimension."""
    objective = pypesto.Objective()

    # define problem with bounds including fixed parameters
    pypesto.Problem(
        objective=objective, lb=[-1] * 4, ub=[1] * 4, dim_full=4,
        x_fixed_indices=[0, 3], x_fixed_vals=[42, 43]
    )

    # define problem with incomplete bounds
    pypesto.Problem(
        objective=objective, lb=[-1] * 2, ub=[1] * 2, dim_full=4,
        x_fixed_indices=[0, 3], x_fixed_vals=[42, 43]
    )


def test_fix_parameters(problem):
    """Test the dynamic problem parameter fixing functions."""
    problem.fix_parameters(2, 45)
    assert problem.x_fixed_indices == [0, 1, 5, 2]
    assert problem.x_fixed_vals == [42, 43, 44, 45]
    assert problem.x_free_indices == [3, 4, 6, 7, 8, 9]

    problem.unfix_parameters(2)
    assert problem.x_fixed_indices == [0, 1, 5]
    assert problem.x_fixed_vals == [42, 43, 44]
    assert problem.x_free_indices == [2, 3, 4, 6, 7, 8, 9]

    # [0, 1, 5] were already fixed, but values should be changed, 5 keeps value
    problem.fix_parameters(range(5), range(5))
    assert problem.x_fixed_indices == [0, 1, 5, 2, 3, 4]
    assert problem.x_fixed_vals == [0, 1, 44, 2, 3, 4]

    with pytest.raises(ValueError):
        problem.fix_parameters(3.5, 2)

    with pytest.raises(ValueError):
        problem.fix_parameters(1, '2')


def test_full_index_to_free_index(problem):
    """Test problem.full_index_to_free_index."""
    assert problem.full_index_to_free_index(2) == 0
    assert problem.full_index_to_free_index(6) == 3
    with pytest.raises(ValueError):
        problem.full_index_to_free_index(5)
