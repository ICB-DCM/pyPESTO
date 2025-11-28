"""Deprecated. Use ``petab_select.plot`` methods instead."""

import warnings

import matplotlib.pyplot as plt
import petab_select.plot
from petab_select import Criterion, Model, Models

from .. import select as pypesto_select


def default_label_maker(model: Model) -> str:
    """Create a model label, for plotting."""
    return str(model.hash)[:4]


# FIXME supply the problem to automatically detect the correct criterion?
def plot_selected_models(
    selected_models: list[Model],
    criterion: str = Criterion.AIC,
    relative: bool = True,
    fz: int = 14,
    size: tuple[float, float] = (5, 4),
    labels: dict[str, str] = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Use `petab_select.plot.line_best_by_iteration`` instead. Deprecated."""
    warnings.warn(
        "Use `petab_select.plot.line_best_by_iteration` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return petab_select.plot.line_best_by_iteration(
        models=Models(selected_models),
        criterion=criterion,
        relative=relative,
        fz=fz,
        labels=labels,
        ax=ax,
    )


def plot_calibrated_models_digraph(
    problem: pypesto_select.Problem,
    calibrated_models: dict[str, Model] = None,
    criterion: Criterion = None,
    optimal_distance: float = 1,
    relative: bool = True,
    options: dict = None,
    labels: dict[str, str] = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Use `petab_select.plot.graph_history`` instead. Deprecated."""
    warnings.warn(
        "Use `petab_select.plot.graph_history` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return petab_select.plot.graph_history(
        models=calibrated_models.values(),
        criterion=criterion,
        draw_networkx_kwargs=options,
        relative=relative,
        labels=labels,
        ax=ax,
    )
