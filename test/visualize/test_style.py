import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pypesto.visualize._style import (
    _DEFAULTS,
    add_colorbar,
    apply_style,
    resolve_style,
)


def test_resolve_style_defaults():
    style = resolve_style()
    assert style == _DEFAULTS


def test_resolve_style_override():
    style = resolve_style({"scatter_size": 99})
    assert style["scatter_size"] == 99
    # other keys stay at default
    assert style["scatter_alpha"] == _DEFAULTS["scatter_alpha"]

    style = resolve_style({"mcmc_scatter_alpha": 0.25})
    assert style["mcmc_scatter_alpha"] == 0.25
    assert "mcmc_trace_alpha" not in style
    assert "mcmc_burnin_alpha" not in style

    style = resolve_style({"rectangle_color": "0.4"})
    assert style["rectangle_color"] == "0.4"

    style = resolve_style({"data_color": "0.1"})
    assert style["data_color"] == "0.1"

    style = resolve_style({"line_color": "0.3"})
    assert style["line_color"] == "0.3"

    style = resolve_style({"cmap_discrete": "plasma"})
    assert style["cmap_discrete"] == "plasma"


def test_resolve_style_unknown_key_warns():
    with pytest.warns(UserWarning, match="Unknown style_kwargs keys"):
        resolve_style({"not_a_key": 1})


def test_add_colorbar():
    fig, axes = plt.subplots(1, 2)
    axes = np.array([[axes[0], axes[1]]])
    values = np.array([0.0, 1.0, 2.0])
    cbar = add_colorbar(fig, axes, values, label="Test label")
    assert cbar.ax.get_ylabel() == "Test label"
    assert cbar.ax.yaxis.label.get_rotation() == 90
    plt.close(fig)


def test_apply_style_keeps_default_font_weights():
    with matplotlib.rc_context():
        apply_style()
        assert (
            matplotlib.rcParams["legend.fontsize"]
            == matplotlib.rcParamsDefault["legend.fontsize"]
        )
        assert (
            matplotlib.rcParams["axes.labelweight"]
            == matplotlib.rcParamsDefault["axes.labelweight"]
        )
