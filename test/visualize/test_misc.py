"""Tests for visualize utility helpers in :mod:`pypesto.visualize.misc`."""

import matplotlib.pyplot as plt
import pytest

from pypesto.visualize.misc import (
    _UNSET,
    get_ax,
    get_axes_array,
    hide_unused_axes,
    process_deprecated_kwarg,
)

from ..conftest import close_fig


@close_fig
def test_get_ax():
    """Returns the given Axes; otherwise creates one with ``size``."""
    _, given = plt.subplots()
    assert get_ax(given) is given

    custom = get_ax(size=(4.0, 3.0))
    assert tuple(custom.get_figure().get_size_inches()) == (4.0, 3.0)


@close_fig
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


@close_fig
def test_hide_unused_axes():
    """Hides unused panels and re-shows reused ones."""
    _, axes = plt.subplots(2, 2, squeeze=False)

    axes = hide_unused_axes(axes=axes, n_used=3, clear=True)
    assert axes[0, 0].get_visible()
    assert axes[1, 0].get_visible()
    assert axes[1, 1].get_visible() is False

    axes = hide_unused_axes(axes=axes, used_indices=(0, 3))
    assert axes[0, 0].get_visible()
    assert axes[0, 1].get_visible() is False
    assert axes[1, 1].get_visible()

    with pytest.raises(ValueError, match="exactly one"):
        hide_unused_axes(axes=axes)


def test_process_deprecated_kwarg():
    """Resolves rename: canonical wins, deprecated warns, both raises."""
    # deprecated not passed (_UNSET) → return canonical
    assert process_deprecated_kwarg("new", 1, "old", _UNSET) == 1
    assert process_deprecated_kwarg("new", None, "old", _UNSET) is None

    # explicit old=None still warns (distinguishable from "not passed")
    with pytest.warns(DeprecationWarning, match="old.*deprecated.*new"):
        assert process_deprecated_kwarg("new", None, "old", None) is None

    with pytest.warns(DeprecationWarning, match="old.*deprecated.*new"):
        assert process_deprecated_kwarg("new", None, "old", 2) == 2

    with pytest.raises(ValueError, match="not both"):
        process_deprecated_kwarg("new", 1, "old", 2)
