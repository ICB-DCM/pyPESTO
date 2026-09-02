"""Tests for :mod:`pypesto.petab.util`."""

import numpy as np
import pandas as pd
import pytest

from pypesto.petab.util import get_petab_v2_extra_field


class Element:
    """Stand-in for a PEtab v2 table element carrying extra fields."""

    def __init__(self, model_extra):
        self.model_extra = model_extra


def _field(value):
    return get_petab_v2_extra_field(
        Element({"parameterType": value}), "parameterType"
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        float("nan"),
        np.float64("nan"),
        np.float32("nan"),
        pd.NA,
        pd.NaT,
        "",
        "   ",
    ],
)
def test_get_petab_v2_extra_field_empty(value):
    """`None`, any null and a blank string all count as unset."""
    assert _field(value) is None


def test_get_petab_v2_extra_field_set():
    """A real value is returned unchanged."""
    assert _field("scaling") == "scaling"


def test_get_petab_v2_extra_field_absent():
    """A missing field, or no extra fields at all, counts as unset."""
    assert get_petab_v2_extra_field(Element({}), "parameterType") is None
    assert get_petab_v2_extra_field(Element(None), "parameterType") is None


@pytest.mark.parametrize(
    "value", [["a", "b"], ("a", "b"), {"a"}, {"a": 1}, np.array([1.0, 2.0])]
)
def test_get_petab_v2_extra_field_rejects_non_scalar(value):
    """A non-scalar value is rejected here rather than deferred.

    Passing it on would surface much later as an opaque
    "Unknown inner parameter type" from `InnerParameterType`.
    """
    with pytest.raises(ValueError, match="Expected a scalar value"):
        _field(value)
