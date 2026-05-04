"""Tests for :mod:`pypesto.visualize._style`."""

import matplotlib.pyplot as plt
import pytest

from pypesto.visualize._style import (
    get_ax,
    get_axes_array,
    process_deprecated_kwarg,
)


def test_get_ax():
    """Returns the given Axes; otherwise creates one with ``size``."""
    _, given = plt.subplots()
    assert get_ax(given) is given

    custom = get_ax(size=(4.0, 3.0))
    assert tuple(custom.get_figure().get_size_inches()) == (4.0, 3.0)

    plt.close("all")


def test_get_axes_array():
    """Normalizes existing grids and creates new ones with ``size``."""
    _, given = plt.subplots(1, 2)
    normalized = get_axes_array(given, nrows=1, ncols=2)
    assert normalized.shape == (1, 2)

    created = get_axes_array(nrows=2, ncols=1, size=(4.0, 3.0))
    assert created.shape == (2, 1)
    assert tuple(created.flat[0].figure.get_size_inches()) == (4.0, 3.0)

    with pytest.raises(ValueError, match="shape"):
        get_axes_array(given, nrows=2, ncols=2)

    plt.close("all")


def test_process_deprecated_kwarg():
    """Resolves rename: canonical wins, deprecated warns, both raises."""
    assert process_deprecated_kwarg("new", 1, "old", None) == 1
    assert process_deprecated_kwarg("new", None, "old", None) is None

    with pytest.warns(DeprecationWarning, match="old.*deprecated.*new"):
        assert process_deprecated_kwarg("new", None, "old", 2) == 2

    with pytest.raises(ValueError, match="not both"):
        process_deprecated_kwarg("new", 1, "old", 2)
