"""Tests for :mod:`pypesto.visualize._style`."""

import matplotlib.pyplot as plt
import pytest

from pypesto.visualize._style import get_ax, process_deprecated_kwarg


def test_get_ax():
    """Returns the given Axes; otherwise creates one with ``size``."""
    _, given = plt.subplots()
    assert get_ax(given) is given

    custom = get_ax(size=(4.0, 3.0))
    assert tuple(custom.get_figure().get_size_inches()) == (4.0, 3.0)

    plt.close("all")


def test_process_deprecated_kwarg():
    """Resolves rename: canonical wins, deprecated warns, both raises."""
    assert process_deprecated_kwarg("new", 1, "old", None) == 1
    assert process_deprecated_kwarg("new", None, "old", None) is None

    with pytest.warns(DeprecationWarning, match="old.*deprecated.*new"):
        assert process_deprecated_kwarg("new", None, "old", 2) == 2

    with pytest.raises(ValueError, match="not both"):
        process_deprecated_kwarg("new", 1, "old", 2)
