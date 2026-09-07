"""Tests for hierarchical optimization with PEtab v2 problems."""

import copy

import pytest

from pypesto.C import MEASUREMENT_TYPE, ORDINAL
from pypesto.hierarchical.petab import validate_hierarchical_petab_problem
from pypesto.testing.examples import (
    get_Boehm_JProteomeRes2014_hierarchical_petab_v2,
)


@pytest.fixture(scope="module")
def petab_problem_v2():
    return get_Boehm_JProteomeRes2014_hierarchical_petab_v2()


def test_hierarchical_petab_v2_validation(petab_problem_v2):
    """Invalid hierarchical PEtab v2 problems are rejected."""
    # the unmodified problem is valid
    validate_hierarchical_petab_problem(petab_problem_v2)

    # unknown parameter type
    petab_problem = copy.deepcopy(petab_problem_v2)
    petab_problem.parameters[-1].model_extra["parameterType"] = "pink"
    with pytest.raises(ValueError, match="Unknown inner parameter type"):
        validate_hierarchical_petab_problem(petab_problem)

    # non-quantitative data types other than relative are not supported yet
    petab_problem = copy.deepcopy(petab_problem_v2)
    petab_problem.measurements[0].model_extra[MEASUREMENT_TYPE] = ORDINAL
    with pytest.raises(NotImplementedError, match="not yet supported"):
        validate_hierarchical_petab_problem(petab_problem)

    # an offset parameter must appear additively in the observable formula
    petab_problem = copy.deepcopy(petab_problem_v2)
    observable = petab_problem.observables[0]
    observable.formula = (
        f"observableParameter1_{observable.id}"
        f" * observableParameter2_{observable.id}"
    )
    with pytest.raises(
        ValueError, match="An offset is in the observable formula"
    ):
        validate_hierarchical_petab_problem(petab_problem)

    # a sigma parameter must constitute the full noise formula
    petab_problem = copy.deepcopy(petab_problem_v2)
    observable = petab_problem.observables[0]
    observable.noise_formula = f"2 * noiseParameter1_{observable.id}"
    with pytest.raises(ValueError, match="full noise formula"):
        validate_hierarchical_petab_problem(petab_problem)

    # non-Gaussian noise distributions are not supported
    petab_problem = copy.deepcopy(petab_problem_v2)
    petab_problem.observables[0].noise_distribution = "laplace"
    with pytest.raises(NotImplementedError, match="oise distribution"):
        validate_hierarchical_petab_problem(petab_problem)
