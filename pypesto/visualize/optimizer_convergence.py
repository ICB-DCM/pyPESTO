from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..C import LABEL_OBJECTIVE
from ..result import Result
from .misc import get_ax
from ._style import resolve_style


def optimizer_convergence(
    result: Result,
    ax: matplotlib.axes.Axes | None = None,
    xscale: str = "symlog",
    yscale: str = "log",
    size: tuple[float, float] | None = None,
    title: str | None = "Optimizer convergence",
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
    """
    Visualize to help spotting convergence issues.

    Scatter plot of function values and gradient norms at the end of
    optimization. Optimizer exit message is encoded by color. Can help
    identifying convergence issues and guide tolerance refinement.

    Parameters
    ----------
    result:
        Optimization result obtained by 'optimize.py'.
    ax:
        Axes object to use.
    xscale:
        Scale for the x-axis (default ``"symlog"``).
    yscale:
        Scale for the y-axis (default ``"log"``).
    size:
        Figure size (width, height) in inches. Only applied when no ``ax``
        is specified.
    title:
        Axes title. Pass ``None`` to suppress.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``scatter_size``, ``scatter_alpha``, ``scatter_linewidths``,
          ``scatter_edgecolors``, ``scatter_zorder`` — scatter point geometry.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    ax:
        The plot axes.
    """
    style = resolve_style(style_kwargs)
    ax = get_ax(ax, size)

    fvals = result.optimize_result.fval
    grad_norms = [
        (
            np.linalg.norm(
                result.problem.get_reduced_vector(
                    grad, result.problem.x_free_indices
                ),
                2,
            )
            if grad is not None
            else np.nan
        )
        for grad in result.optimize_result.grad
    ]
    msgs = result.optimize_result.message

    # Group points by exit message, then scatter once per category.
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    groups: dict[str, tuple[list, list]] = {}
    for fval, grad_norm, msg in zip(fvals, grad_norms, msgs):
        if msg not in groups:
            groups[msg] = ([], [])
        groups[msg][0].append(fval)
        groups[msg][1].append(grad_norm)

    colors = {msg: color_cycle[i % len(color_cycle)] for i, msg in enumerate(groups)}
    for msg, (x_vals, y_vals) in groups.items():
        ax.scatter(
            x_vals, y_vals,
            color=colors[msg],
            s=style["scatter_size"],
            alpha=style["scatter_alpha"],
            linewidths=style["scatter_linewidths"],
            edgecolors=style["scatter_edgecolors"],
            zorder=style["scatter_zorder"],
        )

    # Build one legend handle per category.
    handles = [
        Line2D(
            [0], [0],
            marker="o", color="none",
            markerfacecolor=colors[msg],
            markeredgecolor=style["scatter_edgecolors"],
            markeredgewidth=style["scatter_linewidths"],
            markersize=np.sqrt(style["scatter_size"]),
            label=msg,
        )
        for msg in groups
    ]
    ax.legend(handles=handles, title="Exit message")

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(LABEL_OBJECTIVE)
    ax.set_ylabel("Gradient norm")
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.margins(x=0.05, y=0.05)
    return ax
