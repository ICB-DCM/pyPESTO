from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Literal

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D

from ..C import (
    CONDITION,
    LABEL_LOGPOSTERIOR,
    LEN_RGB,
    MEDIAN,
    OUTPUT,
    RGB,
    STANDARD_DEVIATION,
)
from ..ensemble import EnsemblePrediction, get_percentile_label
from ..problem import Problem
from ..result import McmcPtResult, PredictionResult, Result
from ..sample import calculate_ci_mcmc_sample
from ._style import (
    BOUND_VIEW_MARGIN,
    CI_BAR_HEIGHT,
    COLORBAR_WIDTH,
    GRID_SIZE_PER_COL,
    GRID_SIZE_PER_ROW,
    add_colorbar,
    draw_bounds_1d,
    draw_bounds_2d,
    format_parameter_axis_labels,
    resolve_style,
)
from .misc import (
    _UNSET,
    _ci_panel_lowlevel,
    ci_panel_size,
    get_ax,
    get_axes_array,
    hide_unused_axes,
    make_grid_shape,
    plot_density_panel,
    plot_diagonal_marginal,
    process_deprecated_kwarg,
    rgba2rgb,
)

logger = logging.getLogger(__name__)


def sampling_fval_traces(
    result: Result,
    i_chain: int = 0,
    full_trace: bool = False,
    stepsize: int = 1,
    title: str | None = "Log-posterior trace",
    size: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot log-posterior (=function value) over iterations.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    full_trace:
        Plot the full trace including warm up. Default: False.
    stepsize:
        Only one in `stepsize` values is plotted.
    title:
        Axes title. Pass ``None`` to suppress.
    size:
        Figure size ``(width, height)`` in inches; only used when ``ax`` is
        ``None``.
    ax:
        Axes object to use.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``scatter_size`` — marker area; trace markers are drawn at half
          this value.
        - ``scatter_color`` — color of the posterior-sample points.
        - ``scatter_zorder`` — z-order of the posterior-sample layer.
        - ``mcmc_scatter_alpha`` — MCMC sample marker opacity.
        - ``mcmc_burnin_color`` — burn-in marker color.
        - ``mcmc_burnin_cutoff_color`` — burn-in cutoff line color.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    ax:
        The plot axes.
    """
    style = resolve_style(style_kwargs)

    # get data which should be plotted
    _, params_fval, _, _, _ = get_data_to_plot(
        result=result,
        i_chain=i_chain,
        stepsize=stepsize,
        full_trace=full_trace,
    )

    ax = get_ax(ax, size)

    burn_in = result.sample_result.burn_in or 0

    iterations = params_fval["iteration"].to_numpy()
    log_posterior = params_fval["logPosterior"].to_numpy()
    if full_trace and burn_in > 0:
        is_burnin = iterations < burn_in
    else:
        is_burnin = np.zeros_like(iterations, dtype=bool)

    if is_burnin.any():
        ax.scatter(
            iterations[is_burnin],
            log_posterior[is_burnin],
            color=style["mcmc_burnin_color"],
            alpha=style["mcmc_scatter_alpha"],
            s=style["scatter_size"] / 2,
            linewidths=0.0,
            zorder=2,
            label="Burn-in samples",
        )
    ax.scatter(
        iterations[~is_burnin],
        log_posterior[~is_burnin],
        color=style["scatter_color"],
        alpha=style["mcmc_scatter_alpha"],
        s=style["scatter_size"] / 2,
        linewidths=0.0,
        zorder=style["scatter_zorder"],
        label="Posterior samples",
    )

    if full_trace and burn_in > 0:
        ax.axvline(
            burn_in,
            linestyle="--",
            linewidth=style["bound_linewidth"],
            color=style["mcmc_burnin_cutoff_color"],
            alpha=style["bound_alpha"],
            label="Burn-in cutoff",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(LABEL_LOGPOSTERIOR)

    if title is not None:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=labels)

    return ax


def _get_level_percentiles(level: float) -> tuple[float, float]:
    """Convert a credibility level to percentiles.

    Similar to the highest-density region of a symmetric, unimodal distribution
    (e.g. Gaussian distribution).

    For example, an credibility level of `95` will be converted to
    `(2.5, 97.5)`.

    Parameters
    ----------
    level:
        The credibility level used to calculate the percentiles. For example,
        `[95]` for a 95% credibility interval. These levels are split
        symmetrically, e.g. `95` corresponds to plotting values between the
        2.5% and 97.5% percentiles, and are equivalent to highest-density
        regions for a normal distribution. For skewed distributions, asymmetric
        percentiles may be preferable, but are not yet implemented.

    Returns
    -------
    The percentiles, with the lower percentile first.
    """
    lower_percentile = (100 - level) / 2
    return lower_percentile, 100 - lower_percentile


def _format_percent(level: float) -> str:
    """Format a percentage without a noisy trailing decimal."""
    return f"{level:g}%"


def _normalize_prediction_confidence_levels(
    confidence_levels: float | Sequence[float] | None,
) -> tuple[list[float], list[float]]:
    """Return confidence levels in 0-1 units and percentage units."""
    if confidence_levels is None:
        confidence_levels = [0.95]
    elif isinstance(confidence_levels, (int, float)):
        confidence_levels = [float(confidence_levels)]
    else:
        confidence_levels = [float(level) for level in confidence_levels]

    if any(level > 1 for level in confidence_levels):
        warnings.warn(
            "`confidence_levels` should use 0-1 units "
            "(e.g. `confidence_levels=[0.95]`). Values larger than 1 "
            "are interpreted as percentages and divided by 100.",
            DeprecationWarning,
            stacklevel=3,
        )
        confidence_levels = [
            level / 100 if level > 1 else level
            for level in confidence_levels
        ]

    if any(level <= 0 or level > 1 for level in confidence_levels):
        raise ValueError("`confidence_levels` must be in the interval (0, 1].")

    confidence_levels = sorted(confidence_levels, reverse=True)
    confidence_percentages = [100 * level for level in confidence_levels]
    return confidence_levels, confidence_percentages


def _measurements_from_problem(
    problem: Problem,
    condition_ids: Sequence[str],
    output_ids: Sequence[str],
) -> dict[tuple[str, str], list[np.ndarray]]:
    """Extract per-(condition, output) measurements from an AMICI problem.

    The mapping between ``condition_ids`` (in the order returned by the
    ensemble prediction) and ``problem.objective.edatas`` is positional.
    Missing values (NaN in ``edata.get_measurements()``) are filtered out.

    Returns an empty mapping when ``problem.objective`` is not an
    :class:`~pypesto.objective.AmiciObjective`; a warning is emitted to make
    the silent skip discoverable.
    """
    from ..objective import AmiciObjective

    if not isinstance(problem.objective, AmiciObjective):
        warnings.warn(
            "Cannot extract measurements: `problem.objective` is not an "
            "AmiciObjective. Skipping measurement overlay.",
            stacklevel=3,
        )
        return {}

    edatas = problem.objective.edatas
    observable_ids = list(
        problem.objective.amici_model.get_observable_ids()
    )
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for cond_idx, condition_id in enumerate(condition_ids):
        if cond_idx >= len(edatas):
            break
        edata = edatas[cond_idx]
        timepoints = np.asarray(edata.get_timepoints())
        observed = np.asarray(edata.get_measurements()).reshape(
            len(timepoints), len(observable_ids)
        )
        for output_id in output_ids:
            if output_id not in observable_ids:
                continue
            obs_idx = observable_ids.index(output_id)
            values = observed[:, obs_idx]
            valid = ~np.isnan(values)
            grouped[(condition_id, output_id)] = [
                timepoints[valid],
                values[valid],
            ]
    return grouped


def _prediction_errorbar_settings(style: dict) -> dict:
    """Return errorbar styling consistent with prediction trajectory lines."""
    return {
        "fmt": "none",
        "color": style["line_color"],
        "capsize": 2.0 * style["line_marker_size"],
        "elinewidth": style["trace_linewidth"],
        "capthick": style["marker_linewidth"],
    }


def _get_statistic_data(
    summary: dict[str, PredictionResult],
    statistic: str,
    condition_id: str,
    output_id: str,
) -> tuple[Sequence[float], Sequence[float]]:
    """Get statistic-, condition-, and output-specific data.

    Parameters
    ----------
    summary:
        A `pypesto.ensemble.EnsemblePrediction.prediction_summary`, used as the
        source of annotated data to subset.
    statistic:
        Select data for a specific statistic by its label, e.g. `MEDIAN` or
        `get_percentile_label(95)`.
    condition_id:
        Select data for a specific condition by its ID.
    output_id:
        Select data for a specific output by its ID.

    Returns
    -------
    Predicted values and their corresponding time points. A tuple of two
    sequences, where the first sequence is time points, and the second
    sequence is predicted values at the corresponding time points.
    """
    condition_index = summary[statistic].condition_ids.index(condition_id)
    condition_result = summary[statistic].conditions[condition_index]
    t = condition_result.timepoints
    output_index = condition_result.output_ids.index(output_id)
    y = condition_result.output[:, output_index]
    return (t, y)


def _plot_trajectories_by_condition(
    summary: dict[str, PredictionResult],
    condition_ids: Sequence[str],
    output_ids: Sequence[str],
    axes: matplotlib.axes.Axes,
    levels: Sequence[float],
    level_opacities: Sequence[float],
    labels: dict[str, str],
    variable_colors: Sequence[RGB],
    average: str = MEDIAN,
    add_sd: bool = False,
    grouped_measurements: dict[
        tuple[str, str], Sequence[Sequence[float]]
    ]
    | None = None,
    style: dict | None = None,
) -> None:
    """Plot predicted trajectories, with subplots grouped by condition.

    Parameters
    ----------
    summary:
        A `pypesto.ensemble.EnsemblePrediction.prediction_summary`, used as the
        source of annotated data to plot.
    condition_ids:
        The IDs of conditions to plot.
    output_ids:
        The IDs of outputs to plot.
    axes:
        The axes to plot with. Should contain at least `len(condition_ids)`
        subplots.
    levels:
        Confidence levels as percentages, e.g. [95] for a 95% interval. See
        :py:func:`_get_level_percentiles` for a description of how these
        levels are handled.
    level_opacities:
        A mapping from the confidence levels to the opacities that they should
        be plotted with. Opacity is the only thing that differentiates
        confidence levels in the resulting plot.
    labels:
        Keys should be ensemble output IDs, values should be the desired
        label for that output. Defaults to output IDs.
    variable_colors:
        Colors used to differentiate plotted outputs. The order should
        correspond to `output_ids`.
    average:
        The ID of the statistic that will be plotted as the average (e.g.,
        `MEDIAN` or `MEAN`).
    add_sd:
        Whether to add the standard deviation of the predictions to the plot.
    grouped_measurements:
        Measurement data that has already been grouped by condition and output,
        where the keys are `(condition_id, output_id)` 2-tuples, and the values
        are `[sequence of x-axis values, sequence of y-axis values]`.
    """
    style = resolve_style(style)
    errorbar_settings = _prediction_errorbar_settings(style)
    if grouped_measurements is None:
        grouped_measurements = {}

    # Each subplot has all data for a single condition.
    for condition_index, condition_id in enumerate(condition_ids):
        ax = axes.flat[condition_index]
        ax.set_title(labels[condition_id])
        # Each subplot has all data for all condition-specific outputs.
        for output_index, output_id in enumerate(output_ids):
            # Plot the average for each output.
            t_average, y_average = _get_statistic_data(
                summary,
                average,
                condition_id,
                output_id,
            )
            ax.plot(
                t_average,
                y_average,
                color=style["line_color"],
                linewidth=style["trace_linewidth"],
            )
            if add_sd:
                t_std, y_std = _get_statistic_data(
                    summary,
                    STANDARD_DEVIATION,
                    condition_id,
                    output_id,
                )
                if (t_std != t_average).all():
                    raise ValueError(
                        "Unknown error: timepoints for average and standard "
                        "deviation do not match."
                    )
                ax.errorbar(
                    t_average,
                    y_average,
                    yerr=y_std,
                    **errorbar_settings,
                )
            # Plot the regions described by the confidence level,
            # for each output.
            for level_index, level in enumerate(levels):
                # Get the percentiles that correspond to the confidence level,
                # as their labels in the `summary`.
                lower_label, upper_label = (
                    get_percentile_label(percentile)
                    for percentile in _get_level_percentiles(level)
                )
                # Get the data for each percentile.
                t_lower, lower_data = _get_statistic_data(
                    summary,
                    lower_label,
                    condition_id,
                    output_id,
                )
                t_upper, upper_data = _get_statistic_data(
                    summary,
                    upper_label,
                    condition_id,
                    output_id,
                )
                # Timepoints must match, or `upper_data` will be plotted at
                # some incorrect time points.
                if not (np.array(t_lower) == np.array(t_upper)).all():
                    raise ValueError(
                        "The timepoints of the data for the upper and lower "
                        "percentiles do not match."
                    )
                # Plot a shaded region between the data that correspond to the
                # lower and upper percentiles.
                ax.fill_between(
                    t_lower,
                    lower_data,
                    upper_data,
                    facecolor=rgba2rgb(
                        variable_colors[output_index]
                        + [level_opacities[level_index]]
                    ),
                    lw=0,
                )
            if measurements := grouped_measurements.get(
                (condition_id, output_id), False
            ):
                ax.scatter(
                    measurements[0],
                    measurements[1],
                    marker="s",
                    color=variable_colors[output_index][:LEN_RGB],
                    alpha=style["scatter_alpha"],
                    s=style["scatter_size"],
                    linewidths=style["scatter_linewidths"],
                    edgecolors=style["scatter_edgecolors"],
                    zorder=style["scatter_zorder"],
                )


def _plot_trajectories_by_output(
    summary: dict[str, PredictionResult],
    condition_ids: Sequence[str],
    output_ids: Sequence[str],
    axes: matplotlib.axes.Axes,
    levels: Sequence[float],
    level_opacities: Sequence[float],
    labels: dict[str, str],
    variable_colors: Sequence[RGB],
    average: str = MEDIAN,
    add_sd: bool = False,
    grouped_measurements: dict[
        tuple[str, str], Sequence[Sequence[float]]
    ]
    | None = None,
    condition_gap: float = 0.0,
    style: dict | None = None,
) -> None:
    """Plot predicted trajectories, with subplots grouped by output.

    Each subplot is further divided by conditions, such that all conditions
    are displayed side-by-side for a single output. Hence, in each subplot, the
    timepoints of each condition plot are shifted by the the end timepoint of
    the previous condition plot. For examples of this, see the plots with
    `groupby=OUTPUT` in the example notebook
    `doc/example/sampling_diagnostics.ipynb`.

    See :py:func:`_plot_trajectories_by_condition` for parameter descriptions.
    ``condition_gap`` is added between consecutive condition blocks on the
    cumulative time axis.
    """
    style = resolve_style(style)
    errorbar_settings = _prediction_errorbar_settings(style)
    if grouped_measurements is None:
        grouped_measurements = {}

    # Each subplot has all data for a single output.
    for output_index, output_id in enumerate(output_ids):
        # Store the end timepoint of the previous condition plot, such that the
        # next condition plot starts at the end of the previous condition plot.
        t0 = 0
        ax = axes.flat[output_index]
        ax.set_title(labels[output_id])
        # Collect (abs_start, raw_timepoints, condition_label) for tick relabeling.
        segment_info: list[tuple[float, np.ndarray, str]] = []
        # Each subplot is divided by conditions, with vertical lines.
        for condition_index, condition_id in enumerate(condition_ids):
            seg_start = t0
            facecolor0 = variable_colors[condition_index]
            if condition_index != 0:
                ax.axvline(
                    t0,
                    linewidth=style["trace_linewidth"],
                    color=style["ref_line_color"],
                )

            t_max = t0
            t_average, y_average = _get_statistic_data(
                summary,
                average,
                condition_id,
                output_id,
            )
            # Shift the timepoints for the average plot to start at the end of
            # the previous condition plot.
            t_average_shifted = t_average + t0
            ax.plot(
                t_average_shifted,
                y_average,
                color=style["line_color"],
                linewidth=style["trace_linewidth"],
            )
            if add_sd:
                t_std, y_std = _get_statistic_data(
                    summary,
                    STANDARD_DEVIATION,
                    condition_id,
                    output_id,
                )
                if (t_std != t_average).all():
                    raise ValueError(
                        "Unknown error: timepoints for average and standard "
                        "deviation do not match."
                    )
                ax.errorbar(
                    t_average_shifted,
                    y_average,
                    yerr=y_std,
                    **errorbar_settings,
                )
            t_max = max(t_max, *t_average_shifted)
            for level_index, level in enumerate(levels):
                # Get the percentiles that correspond to the confidence level,
                # as their labels in the `summary`.
                lower_label, upper_label = (
                    get_percentile_label(percentile)
                    for percentile in _get_level_percentiles(level)
                )
                # Get the data for each percentile.
                t_lower, lower_data = _get_statistic_data(
                    summary,
                    lower_label,
                    condition_id,
                    output_id,
                )
                t_upper, upper_data = _get_statistic_data(
                    summary,
                    upper_label,
                    condition_id,
                    output_id,
                )
                # Shift the timepoints for the `fill_between` plots to start at
                # the end of the previous condition plot.
                t_lower_shifted = t_lower + t0
                t_upper_shifted = t_upper + t0
                # Timepoints must match, or `upper_data` will be plotted at
                # some incorrect time points.
                if not (np.array(t_lower) == np.array(t_upper)).all():
                    raise ValueError(
                        "The timepoints of the data for the upper and lower "
                        "percentiles do not match."
                    )
                # Plot a shaded region between the data that correspond to the
                # lower and upper percentiles.
                ax.fill_between(
                    t_lower_shifted,
                    lower_data,
                    upper_data,
                    facecolor=rgba2rgb(
                        facecolor0 + [level_opacities[level_index]]
                    ),
                    lw=0,
                )
                t_max = max(t_max, *t_lower_shifted, *t_upper_shifted)
            if measurements := grouped_measurements.get(
                (condition_id, output_id), False
            ):
                ax.scatter(
                    [t0 + _t for _t in measurements[0]],
                    measurements[1],
                    marker="s",
                    color=variable_colors[condition_index][:LEN_RGB],
                    alpha=style["scatter_alpha"],
                    s=style["scatter_size"],
                    linewidths=style["scatter_linewidths"],
                    edgecolors=style["scatter_edgecolors"],
                    zorder=style["scatter_zorder"],
                )
            # Set t0 to the last plotted timepoint of the current condition
            # plot.
            segment_info.append(
                (seg_start, t_average, labels[condition_id], condition_index)
            )
            t0 = t_max + condition_gap

        # Per-segment x-ticks: labels restart at the raw (relative) time for
        # each condition so that the axis reads "0 … t_max" four times rather
        # than showing cumulative values.
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        locator = matplotlib.ticker.MaxNLocator(nbins=3, integer=True)
        last_idx = len(segment_info) - 1
        for seg_idx, (seg_start_i, t_raw, _, _) in enumerate(segment_info):
            t_lo, t_hi = float(t_raw[0]), float(t_raw[-1])
            nice = locator.tick_values(t_lo, t_hi)
            # Drop the right endpoint for all but the last segment to avoid
            # boundary collisions with the next segment's leftmost tick.
            upper_inclusive = seg_idx == last_idx
            mask = (nice >= t_lo) & (
                nice <= t_hi if upper_inclusive else nice < t_hi
            )
            for t_rel in nice[mask]:
                tick_positions.append(float(seg_start_i) + float(t_rel))
                tick_labels.append(
                    f"{t_rel:.0f}" if t_rel % 1 == 0 else f"{t_rel:.2g}"
                )
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        # Condition-name annotations inside the plot near the top of each
        # segment, colour-matched to the condition.
        blended = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes
        )
        for seg_start_i, t_raw, cond_label, cond_idx in segment_info:
            seg_mid = float(seg_start_i) + (float(t_raw[-1]) - float(t_raw[0])) / 2
            ax.text(
                seg_mid,
                0.97,
                cond_label,
                transform=blended,
                ha="center",
                va="top",
                fontsize="x-small",
                color=variable_colors[cond_idx][:3],
                clip_on=True,
            )


def _get_condition_and_output_ids(
    summary: dict[str, PredictionResult],
) -> tuple[Sequence[str], Sequence[str]]:
    """Get all condition and output IDs in a prediction summary.

    Parameters
    ----------
    summary:
        The prediction summary to extract condition and output IDs from.

    Returns
    -------
    A 2-tuple, with the following indices and values.
    - `0`: a list of all condition IDs.
    - `1`: a list of all output IDs.
    """
    # For now, all prediction results must predict for the same set of
    # conditions. Can support different conditions later.
    all_condition_ids = [
        prediction.condition_ids for prediction in summary.values()
    ]
    if not (
        np.array(
            [
                set(condition_ids) == set(all_condition_ids[0])
                for condition_ids in all_condition_ids
            ]
        ).all()
    ):
        raise KeyError("All predictions must have the same set of conditions.")
    condition_ids = all_condition_ids[0]

    output_ids = sorted(
        {
            output_id
            for prediction in summary.values()
            for condition in prediction.conditions
            for output_id in condition.output_ids
        }
    )

    return condition_ids, output_ids


def _handle_legends(
    fig: mpl.figure.Figure,
    confidence_level_percentages: Sequence[float],
    labels: dict[str, str],
    level_opacities: Sequence[float],
    variable_names: Sequence[str],
    variable_colors: Sequence[RGB],
    average: str,
    add_sd: bool,
    grouped_measurements: dict[tuple[str, str], Sequence[Sequence[float]]]
    | None,
    style: dict,
) -> None:
    """Add a single combined figure-level legend below the subplot grid.

    Variables (coloured) fill the upper rows; prediction styles (greyscale)
    occupy the bottom row. The deliberate "outside" placement is an exception
    to pyPESTO's inside-legend convention because legend entries scale with
    the number of observables / conditions, which would crowd in-axes legends.
    Assumes the figure uses matplotlib's constrained layout so the engine
    reserves space below the axes automatically.

    Parameters
    ----------
    fig:
        Target figure for the figure-level legend.
    confidence_level_percentages:
        The confidence levels, expressed as percentages for legend labels.
    labels:
        Display labels for output / condition IDs.
    level_opacities:
        The opacity to plot each confidence level with.
    variable_names:
        The name of each variable.
    variable_colors:
        The color to plot each variable in.
    average:
        The ID of the statistic that will be plotted as the average (e.g.,
        `MEDIAN` or `MEAN`).
    add_sd:
        Whether to add the standard deviation of the predictions to the plot.
    grouped_measurements:
        Measurement data already grouped by condition and output; presence of
        any entries adds a "Data" entry to the prediction-style row.
    style:
        Resolved pyPESTO visualization style.
    """
    fake_data = [[0], [0]]

    # Build variable-colour legend handles (one line per output / condition).
    variable_lines = np.array(
        [
            [
                labels[variable_name],
                Line2D(
                    *fake_data,
                    color=variable_colors[index],
                    linewidth=style["trace_linewidth"],
                ),
            ]
            for index, variable_name in enumerate(variable_names)
        ]
    )

    # Build CI-level legend handles.
    ci_lines = []
    ci_color = list(mpl.colors.to_rgb(style["line_color"]))
    for index, level in enumerate(confidence_level_percentages):
        ci_lines.append(
            [
                f"{_format_percent(level)} CI",
                Line2D(
                    *fake_data,
                    color=rgba2rgb([*ci_color, level_opacities[index]]),
                    linewidth=max(2.0, 2.0 * style["trace_linewidth"]),
                ),
            ]
        )

    average_title = average.title()
    average_line_object_line2d = Line2D(
        *fake_data,
        color=style["line_color"],
        linewidth=style["trace_linewidth"],
    )
    if add_sd:
        capline = Line2D(
            *fake_data,
            color=style["line_color"],
            markersize=4.0 * style["line_marker_size"],
        )
        average_title += " + SD"
        barline = LineCollection(
            np.empty((2, 2, 2)),
            color=style["line_color"],
            linewidth=style["trace_linewidth"],
        )
        average_line_object = ErrorbarContainer(
            (average_line_object_line2d, [capline], [barline]),
            has_yerr=True,
        )
    else:
        average_line_object = average_line_object_line2d
    average_line = [[average_title, average_line_object]]

    data_line = []
    if grouped_measurements:
        data_line = [
            [
                "Data",
                Line2D(
                    *fake_data,
                    linewidth=0,
                    marker="s",
                    markerfacecolor="0.7",
                    markeredgecolor=style["scatter_edgecolors"],
                    markersize=np.sqrt(style["scatter_size"]),
                    markeredgewidth=style["scatter_linewidths"],
                    alpha=style["scatter_alpha"],
                ),
            ]
        ]

    level_lines = np.array(ci_lines + average_line + data_line)

    # Combined legend with a row-major display: variables fill the top rows,
    # prediction styles occupy the bottom row. matplotlib fills legends
    # column-by-column, so handles are interleaved per-column to achieve the
    # desired row-major appearance.
    n_pred = len(level_lines)
    ncol = n_pred
    blank = Line2D([], [], color="none", linewidth=0)

    if len(variable_names) > 1:
        n_vars = len(variable_names)
        n_var_rows = (n_vars + ncol - 1) // ncol
        var_h = list(variable_lines[:, 1]) + [blank] * (
            n_var_rows * ncol - n_vars
        )
        var_l = list(variable_lines[:, 0]) + [""] * (
            n_var_rows * ncol - n_vars
        )
        legend_handles: list = []
        legend_labels: list = []
        for c in range(ncol):
            for r in range(n_var_rows):
                idx = r * ncol + c
                legend_handles.append(var_h[idx])
                legend_labels.append(var_l[idx])
            legend_handles.append(level_lines[c, 1])
            legend_labels.append(level_lines[c, 0])
    else:
        legend_handles = list(level_lines[:, 1])
        legend_labels = list(level_lines[:, 0])

    fig.legend(
        legend_handles,
        legend_labels,
        loc="outside lower center",
        ncol=ncol,
    )


def _handle_colors(
    confidence_levels: Sequence[float],
    n_variables: int,
    style: dict,
    reverse: bool = False,
) -> tuple[Sequence[float], Sequence[RGB]]:
    """Calculate the colors for the prediction trajectories plot.

    Parameters
    ----------
    confidence_levels:
        The confidence levels.
    n_variables:
        The maximum possible number of variables per subplot.
    style:
        Resolved pyPESTO visualization style.

    Returns
    -------
    A 2-tuple, with the following indices and values.
    - `0`: a list of opacities, one per level.
    - `1`: a list of colors, one per variable.
    """
    max_alpha = float(style["ci_alpha"])
    level_opacities = sorted(
        np.linspace(0.35 * max_alpha, max_alpha, len(confidence_levels)),
        reverse=reverse,
    )

    cmap = plt.get_cmap(style["cmap_discrete"])
    # Use endpoint=False so qualitative cmaps (tab10, tab20, …) sample
    # distinct colour bands rather than interpolating towards the same hue.
    variable_colors = [
        list(cmap(v))[:LEN_RGB]
        for v in np.linspace(0, 1, max(n_variables, 1), endpoint=False)
    ]

    return level_opacities, variable_colors


def sampling_prediction_trajectories(
    ensemble_prediction: EnsemblePrediction,
    problem: Problem | None = None,
    confidence_levels: float | Sequence[float] | None = None,
    title: str | None = None,
    size: tuple[float, float] | None = None,
    axes: matplotlib.axes.Axes | np.ndarray | None = None,
    labels: dict[str, str] | None = None,
    groupby: str = CONDITION,
    condition_gap: float = 0.01,
    condition_ids: Sequence[str] | None = None,
    output_ids: Sequence[str] | None = None,
    weighting: bool = False,
    reverse_opacities: bool = False,
    average: str = MEDIAN,
    add_sd: bool = False,
    style_kwargs: dict | None = None,
    axis_label_padding: int = _UNSET,
    levels: float | Sequence[float] = _UNSET,
) -> np.ndarray:
    """
    Visualize prediction trajectory of an EnsemblePrediction.

    Plot MCMC-based prediction confidence intervals for the model states or
    outputs. One or various confidence levels can be depicted.

    Parameters
    ----------
    ensemble_prediction:
        The ensemble prediction.
    problem:
        Optional pyPESTO problem. When provided with an AMICI objective,
        measurement data is overlaid on the trajectories.
    confidence_levels:
        Confidence levels in 0-1 units, e.g. ``[0.95]`` for a 95%
        interval. Levels are split symmetrically; see
        :py:func:`_get_level_percentiles`.
    title:
        Figure title.
    size:
        Figure size ``(width, height)`` in inches; only used when ``axes`` is
        ``None``.
    axes:
        Axes grid to use. Must match the computed subplot layout.
    labels:
        Keys should be ensemble output IDs, values should be the desired
        label for that output. Defaults to output IDs.
    axis_label_padding:
        Deprecated. Has no effect; axis labels are now set via
        ``ax.set_xlabel`` / ``ax.set_ylabel``.
    groupby:
        Group plots by `pypesto.C.OUTPUT` or
        `pypesto.C.CONDITION`.
    condition_gap:
        Gap between conditions when ``groupby == pypesto.C.OUTPUT``.
    condition_ids:
        If provided, only data for the provided condition IDs will be plotted.
    output_ids:
        If provided, only data for the provided output IDs will be plotted.
    weighting:
        Whether weights should be used for trajectory.
    reverse_opacities:
        Whether to reverse the opacities that are assigned to different levels.
    average:
        The ID of the statistic that will be plotted as the average (e.g.,
        `MEDIAN` or `MEAN`).
    add_sd:
        Whether to add the standard deviation of the predictions to the plot.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``cmap_discrete`` — qualitative colormap used to distinguish
          outputs or conditions.
        - ``ci_alpha`` — maximum opacity of the confidence interval bands.
        - ``line_color`` — mean/median trajectory color.
        - ``ref_line_color`` — condition separator color for
          ``groupby="output"``.
        - ``trace_linewidth`` — line width of trajectories, separators, and
          standard-deviation error bars.
        - ``scatter_size``, ``scatter_alpha``, ``scatter_linewidths``,
          ``scatter_edgecolors``, ``scatter_zorder`` — measurement point
          geometry.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.
    levels:
        Deprecated. Use ``confidence_levels`` in 0-1 units instead. Old
        percentage values are converted automatically.

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.
    """
    style = resolve_style(style_kwargs)
    process_deprecated_kwarg(
        canonical_name=None,
        canonical_value=None,
        deprecated_name="axis_label_padding",
        deprecated_value=axis_label_padding,
        note="Axis labels are now set via ax.set_xlabel / ax.set_ylabel.",
    )
    confidence_levels = process_deprecated_kwarg(
        "confidence_levels",
        confidence_levels,
        "levels",
        levels,
    )
    confidence_levels, confidence_level_percentages = (
        _normalize_prediction_confidence_levels(confidence_levels)
    )

    if labels is None:
        labels = {}
    # Get the percentiles that correspond to the requested confidence levels.
    percentiles = [
        percentile
        for level in confidence_level_percentages
        for percentile in _get_level_percentiles(level)
    ]

    summary = ensemble_prediction.compute_summary(
        percentiles_list=percentiles, weighting=weighting
    )

    all_condition_ids, all_output_ids = _get_condition_and_output_ids(summary)
    if condition_ids is None:
        condition_ids = all_condition_ids
    condition_ids = list(condition_ids)
    if output_ids is None:
        output_ids = all_output_ids
    output_ids = list(output_ids)

    # Extract measurements from the AMICI objective if a problem was passed.
    if problem is not None:
        grouped_measurements = _measurements_from_problem(
            problem, condition_ids, output_ids
        )
    else:
        grouped_measurements = {}

    # Set default labels for any unspecified labels.
    labels = {id_: labels.get(id_, id_) for id_ in condition_ids + output_ids}

    if groupby == CONDITION:
        n_variables = len(output_ids)
        variable_names = output_ids
        n_subplots = len(condition_ids)
    elif groupby == OUTPUT:
        n_variables = len(condition_ids)
        variable_names = condition_ids
        n_subplots = len(output_ids)
    else:
        raise ValueError(f"Unsupported groupby value: {groupby}")

    level_opacities, variable_colors = _handle_colors(
        confidence_levels=confidence_levels,
        n_variables=n_variables,
        style=style,
        reverse=reverse_opacities,
    )

    n_row, n_col = make_grid_shape(n_subplots)
    axes = get_axes_array(axes=axes, nrows=n_row, ncols=n_col, size=size)
    fig = axes.flat[0].figure
    axes = hide_unused_axes(axes=axes, n_used=n_subplots, clear=True)

    if groupby == CONDITION:
        _plot_trajectories_by_condition(
            summary=summary,
            condition_ids=condition_ids,
            output_ids=output_ids,
            axes=axes,
            levels=confidence_level_percentages,
            level_opacities=level_opacities,
            labels=labels,
            variable_colors=variable_colors,
            average=average,
            add_sd=add_sd,
            grouped_measurements=grouped_measurements,
            style=style,
        )
    elif groupby == OUTPUT:
        _plot_trajectories_by_output(
            summary=summary,
            condition_ids=condition_ids,
            output_ids=output_ids,
            axes=axes,
            levels=confidence_level_percentages,
            level_opacities=level_opacities,
            labels=labels,
            variable_colors=variable_colors,
            average=average,
            add_sd=add_sd,
            grouped_measurements=grouped_measurements,
            condition_gap=condition_gap,
            style=style,
        )

    if title is not None:
        fig.suptitle(title)

    _handle_legends(
        fig=fig,
        confidence_level_percentages=confidence_level_percentages,
        labels=labels,
        level_opacities=level_opacities,
        variable_names=variable_names,
        variable_colors=variable_colors,
        average=average,
        add_sd=add_sd,
        grouped_measurements=grouped_measurements,
        style=style,
    )

    # Axis labels: x on every visible panel; y on leftmost column only.
    # For groupby=OUTPUT the x-axis shows per-segment time (restarting at 0 for
    # each condition); condition names are annotated directly on the subplots.
    xlabel = "Time"
    for idx, ax in enumerate(axes.flat):
        if not ax.get_visible():
            continue
        ax.set_xlabel(xlabel)
        if idx % n_col == 0:
            ax.set_ylabel("Simulated values")

    return axes


def sampling_parameter_cis(
    result: Result,
    confidence_levels: Sequence[float] | None = None,
    show_median: bool = True,
    show_bounds: bool = True,
    orientation: Literal["v", "h"] = "v",
    size: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = "Sampling credibility intervals",
    alpha: Sequence[int] | None = None,
    step: float = _UNSET,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot MCMC-based parameter credibility intervals.

    Uses :func:`~pypesto.visualize.misc._ci_panel_lowlevel` for rendering,
    sharing the same visual language as :func:`profile_cis`.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    confidence_levels:
        Credibility levels as fractions in (0, 1), e.g. ``[0.95]``.
        Defaults to ``[0.95]``.
    show_median:
        Mark the posterior median with a tick on each CI bar.
    show_bounds:
        Whether to draw parameter bounds.
    orientation:
        ``"v"`` (default): parameters on y-axis, values on x-axis.
        ``"h"``: transposed — parameters on x-axis, values on y-axis.
    size:
        Figure size in inches. Defaults to scaling with number of parameters.
    ax:
        Axes object to use.
    title:
        Axes title. Pass ``None`` to suppress.
    alpha:
        Deprecated. Use ``confidence_levels`` instead.
    step:
        Deprecated. Has no effect; bar heights are now computed automatically.
    style_kwargs:
        Optional style overrides. Supported keys:

        - ``"cmap_ci"`` – colormap for CI bars (default ``"Blues"``).
        - ``"bound_color"``, ``"bound_linestyle"``, ``"bound_linewidth"``,
          ``"bound_alpha"`` – bound line appearance.

    Returns
    -------
    ax:
        The plot axes.
    """
    style = resolve_style(style_kwargs)

    if alpha is not None:
        if confidence_levels is not None:
            raise ValueError(
                "Pass either `confidence_levels` or the deprecated `alpha`, not both."
            )
        warnings.warn(
            "`alpha` is deprecated; use `confidence_levels` instead. "
            "Note: units have changed — pass fractions in (0, 1) "
            "(e.g. `confidence_levels=[0.95]`) instead of integer percentages "
            "(e.g. `alpha=[95]`). Your values have been divided by 100 automatically.",
            DeprecationWarning,
            stacklevel=2,
        )
        confidence_levels = [a / 100 for a in alpha]

    if step is not _UNSET:
        warnings.warn(
            "`step` is deprecated and has no effect. Bar heights are now "
            "determined automatically from the number of confidence levels.",
            DeprecationWarning,
            stacklevel=2,
        )

    if confidence_levels is None:
        confidence_levels = [0.95]

    levels_sorted = sorted([float(cl) for cl in confidence_levels], reverse=True)
    n_cls = len(levels_sorted)
    ws = [(CI_BAR_HEIGHT / n_cls) * i for i in range(1, n_cls + 1)]
    cmap = mpl.colormaps[style["cmap_ci"]]
    colors = [cmap(0.3 + 0.6 * w / max(ws)) for w in ws]

    n_pars = result.sample_result.trace_x.shape[-1]

    # build ci_data: one entry per level, widest first
    ci_data = []
    for level, color in zip(levels_sorted, colors):
        lb, ub = calculate_ci_mcmc_sample(result=result, ci_level=level)
        ci_data.append((level, lb, ub, color))

    # posterior median as point estimate
    point_estimates = None
    if show_median:
        burn_in = result.sample_result.burn_in
        trace = result.sample_result.trace_x[0, burn_in:, :]
        point_estimates = np.median(trace, axis=0)

    lb_full = list(result.problem.get_reduced_vector(result.problem.lb_full))
    ub_full = list(result.problem.get_reduced_vector(result.problem.ub_full))
    x_names = list(result.problem.get_reduced_vector(result.problem.x_names))
    x_scales = (
        list(result.problem.get_reduced_vector(result.problem.x_scales))
        if getattr(result.problem, "x_scales", None) is not None
        else None
    )

    if size is None:
        size = ci_panel_size(n_pars, orientation)
    ax = get_ax(ax, size)

    return _ci_panel_lowlevel(
        ax, ci_data, x_names, x_scales, lb_full, ub_full, style,
        point_estimates=point_estimates,
        point_estimate_label="Posterior median",
        show_bounds=show_bounds,
        title=title,
        legend_title="Credibility level:",
        orientation=orientation,
    )


def sampling_parameter_traces(
    result: Result,
    i_chain: int = 0,
    parameter_indices: Sequence[int] | None = None,
    full_trace: bool = False,
    stepsize: int = 1,
    use_problem_bounds: bool = _UNSET,
    show_bounds: bool | None = None,
    title: str | None = None,
    size: tuple[float, float] | None = None,
    axes: np.ndarray | None = None,
    ax: np.ndarray | None = _UNSET,
    par_indices: Sequence[int] = _UNSET,
    suptitle: str | None = _UNSET,
    style_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Plot parameter values over iterations.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    parameter_indices: list of integer values
        List of integer values specifying which parameters to plot.
        Default: All parameters are shown.
    full_trace:
        Plot the full trace including warm up. Default: False.
    stepsize:
        Only one in `stepsize` values is plotted.
    use_problem_bounds:
        Deprecated. Use ``show_bounds`` instead.
    show_bounds:
        Whether to draw lower and upper parameter bounds and frame the y-axis
        to include them.
    title:
        Figure title.
    size:
        Figure size ``(width, height)`` in inches; only used when ``axes``
        is ``None``.
    axes:
        Axes grid to use. Must match the computed subplot layout.
    ax:
        Deprecated. Use ``axes`` instead.
    par_indices:
        Deprecated. Use ``parameter_indices`` instead.
    suptitle:
        Deprecated. Use ``title`` instead.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``scatter_size`` — marker area; trace markers are drawn at half
          this value.
        - ``scatter_color`` — color of the posterior-sample points.
        - ``scatter_zorder`` — z-order of the posterior-sample layer.
        - ``mcmc_scatter_alpha`` — MCMC sample marker opacity.
        - ``mcmc_burnin_color`` — burn-in marker color.
        - ``mcmc_burnin_cutoff_color`` — burn-in cutoff line color.
        - ``bound_color``, ``bound_linestyle``, ``bound_linewidth``,
          ``bound_alpha`` — parameter-bound line style.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.
    """
    style = resolve_style(style_kwargs)
    title = process_deprecated_kwarg("title", title, "suptitle", suptitle)

    parameter_indices = process_deprecated_kwarg(
        "parameter_indices",
        parameter_indices,
        "par_indices",
        par_indices,
    )
    axes = process_deprecated_kwarg("axes", axes, "ax", ax)
    show_bounds = process_deprecated_kwarg(
        "show_bounds",
        show_bounds,
        "use_problem_bounds",
        use_problem_bounds,
    )
    if show_bounds is None:
        show_bounds = True

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub, param_names = get_data_to_plot(
        result=result,
        i_chain=i_chain,
        stepsize=stepsize,
        full_trace=full_trace,
        parameter_indices=parameter_indices,
    )

    num_row, num_col = make_grid_shape(nr_params)
    axes = get_axes_array(axes=axes, nrows=num_row, ncols=num_col, size=size)
    fig = axes.flat[0].figure
    axes = hide_unused_axes(axes=axes, n_used=nr_params, clear=True)

    par_ax = dict(zip(param_names, axes.flat))
    all_reduced_names = result.problem.get_reduced_vector(result.problem.x_names)
    name_to_reduced_idx = {name: i for i, name in enumerate(all_reduced_names)}

    burn_in = result.sample_result.burn_in or 0
    iterations = params_fval["iteration"].to_numpy()
    if full_trace and burn_in > 0:
        is_burnin = iterations < burn_in
    else:
        is_burnin = np.zeros_like(iterations, dtype=bool)

    for idx, plot_id in enumerate(param_names):
        _ax = par_ax[plot_id]
        values = params_fval[plot_id].to_numpy()

        if is_burnin.any():
            _ax.scatter(
                iterations[is_burnin],
                values[is_burnin],
                color=style["mcmc_burnin_color"],
                alpha=style["mcmc_scatter_alpha"],
                s=style["scatter_size"] / 2,
                linewidths=0.0,
                zorder=2,
                label="Burn-in samples" if idx == 0 else None,
            )
        _ax.scatter(
            iterations[~is_burnin],
            values[~is_burnin],
            color=style["scatter_color"],
            alpha=style["mcmc_scatter_alpha"],
            s=style["scatter_size"] / 2,
            linewidths=0.0,
            zorder=style["scatter_zorder"],
            label="Posterior samples" if idx == 0 else None,
        )

        if full_trace and burn_in > 0:
            _ax.axvline(
                burn_in,
                linestyle="--",
                linewidth=style["bound_linewidth"],
                color=style["mcmc_burnin_cutoff_color"],
                alpha=style["bound_alpha"],
                label="Burn-in cutoff" if idx == 0 else None,
            )

        _ax.set_xlabel("Iteration")
        _ax.set_ylabel(param_names[idx])
        if show_bounds:
            par_reduced_idx = name_to_reduced_idx.get(plot_id, idx)
            draw_bounds_1d(
                _ax,
                float(theta_lb[par_reduced_idx]),
                float(theta_ub[par_reduced_idx]),
                axis="y",
                view_margin=True,
                style=style,
            )
            if idx == 0:
                _ax.plot(
                    [],
                    [],
                    color=style["bound_color"],
                    linestyle=style["bound_linestyle"],
                    linewidth=style["bound_linewidth"],
                    alpha=style["bound_alpha"],
                    label="Bounds",
                )

        if idx == 0:
            handles, labels = _ax.get_legend_handles_labels()
            if handles:
                _ax.legend(handles=handles, labels=labels)

    if title is not None:
        fig.suptitle(title)

    return axes


def sampling_scatter(
    result: Result,
    i_chain: int = 0,
    stepsize: int = 1,
    parameter_indices: Sequence[int] | None = None,
    title: str | None = None,
    diag_kind: str = "kde",
    size: tuple[float, float] | None = None,
    show_bounds: bool = True,
    axes: np.ndarray | None = None,
    suptitle: str | None = _UNSET,
    style_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Parameter scatter plot.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    stepsize:
        Only one in `stepsize` values is plotted.
    parameter_indices:
        Indices of parameters to show. Defaults to all free parameters.
    title:
        Figure title.
    diag_kind:
        Visualization mode for marginal densities {‘auto’, ‘hist’, ‘kde’, None}
    size:
        Figure size in inches.
    show_bounds:
        Whether to show, and extend the plot to, the lower and upper bounds.
    suptitle:
        Deprecated. Use ``title`` instead.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``scatter_size``, ``scatter_alpha``, ``scatter_linewidths``,
          ``scatter_edgecolors``, ``scatter_zorder`` — off-diagonal
          scatter point geometry.
        - ``cmap_posterior`` — colormap used for the log-posterior-encoded
          scatter colours and the colorbar.
        - ``rectangle_color`` — diagonal marginal fill colour.
        - ``bound_color``, ``bound_linestyle``, ``bound_linewidth``,
          ``bound_alpha`` — parameter-bound line style.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.
    """
    style = resolve_style(style_kwargs)
    title = process_deprecated_kwarg("title", title, "suptitle", suptitle)

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub, param_names = get_data_to_plot(
        result=result, i_chain=i_chain, stepsize=stepsize,
        parameter_indices=parameter_indices,
    )

    if size is None and axes is None:
        # grid panels + extra width for the shared colorbar
        size = (
            GRID_SIZE_PER_COL * nr_params + COLORBAR_WIDTH,
            GRID_SIZE_PER_ROW * nr_params,
        )

    axes = get_axes_array(
        axes=axes, nrows=nr_params, ncols=nr_params, size=size
    )
    fig = axes.flat[0].figure
    previous_colorbar_axes = []
    for ax in axes.flat:
        colorbar_ax = getattr(ax, "_pypesto_sampling_scatter_colorbar_ax", None)
        if (
            colorbar_ax is not None
            and colorbar_ax not in previous_colorbar_axes
        ):
            previous_colorbar_axes.append(colorbar_ax)
    for colorbar_ax in previous_colorbar_axes:
        if colorbar_ax in fig.axes:
            colorbar_ax.remove()

    for ax in axes.flat:
        ax.clear()
        ax.set_visible(True)
        if hasattr(ax, "_pypesto_sampling_scatter_colorbar_ax"):
            delattr(ax, "_pypesto_sampling_scatter_colorbar_ax")

    log_posterior_values = params_fval["logPosterior"].to_numpy()

    import matplotlib.colors as mpl_colors

    _cmap = style["cmap_posterior"]
    _norm = mpl_colors.Normalize(
        vmin=float(log_posterior_values.min()),
        vmax=float(log_posterior_values.max()),
    )
    _cmap_obj = plt.get_cmap(_cmap) if isinstance(_cmap, str) else _cmap

    data = params_fval[param_names]
    for row in range(nr_params):
        for col in range(nr_params):
            ax = axes[row, col]
            col_name = param_names[col]
            row_name = param_names[row]
            col_vals = data[col_name]
            row_vals = data[row_name]

            if row == col:
                plot_diagonal_marginal(
                    ax=ax,
                    values=col_vals,
                    diag_kind=diag_kind,
                    color=style["rectangle_color"],
                )
            else:
                ax.scatter(
                    col_vals,
                    row_vals,
                    c=log_posterior_values,
                    cmap=_cmap_obj,
                    norm=_norm,
                    alpha=style["scatter_alpha"],
                    s=style["scatter_size"],
                    linewidths=style["scatter_linewidths"],
                    edgecolors=style["scatter_edgecolors"],
                    zorder=style["scatter_zorder"],
                )
                ax.set_ylabel(row_name)

            ax.set_xlabel(col_name)

    if show_bounds:
        # Compute lim per col/row including data + bounds + view margin
        for col in range(nr_params):
            col_vals = data[param_names[col]].to_numpy()
            lo = min(float(col_vals.min()), float(theta_lb[col]))
            hi = max(float(col_vals.max()), float(theta_ub[col]))
            span = hi - lo
            pad = span * BOUND_VIEW_MARGIN if span > 0 else 0.5
            xlim = (lo - pad, hi + pad)
            for row in range(nr_params):
                axes[row, col].set_xlim(xlim)
        for row in range(nr_params):
            row_vals = data[param_names[row]].to_numpy()
            lo = min(float(row_vals.min()), float(theta_lb[row]))
            hi = max(float(row_vals.max()), float(theta_ub[row]))
            span = hi - lo
            pad = span * BOUND_VIEW_MARGIN if span > 0 else 0.5
            ylim = (lo - pad, hi + pad)
            for col in range(nr_params):
                if row != col:
                    axes[row, col].set_ylim(ylim)
        for row in range(nr_params):
            for col in range(nr_params):
                ax = axes[row, col]
                if row == col:
                    draw_bounds_1d(
                        ax,
                        float(theta_lb[col]),
                        float(theta_ub[col]),
                        axis="x",
                        view_margin=False,
                        style=style,
                    )
                else:
                    draw_bounds_2d(
                        ax,
                        float(theta_lb[col]),
                        float(theta_ub[col]),
                        float(theta_lb[row]),
                        float(theta_ub[row]),
                        view_margin=False,
                        style=style,
                    )

    cbar = add_colorbar(
        fig,
        axes,
        log_posterior_values,
        LABEL_LOGPOSTERIOR,
        cmap=style["cmap_posterior"],
        norm=_norm,
    )
    for ax in axes.flat:
        ax._pypesto_sampling_scatter_colorbar_ax = cbar.ax

    if title is not None:
        fig.suptitle(title)

    return axes


def sampling_1d_marginals(
    result: Result,
    i_chain: int = 0,
    parameter_indices: Sequence[int] | None = None,
    stepsize: int = 1,
    plot_type: str = "both",
    bw_method: str = "scott",
    show_bounds: bool = True,
    title: str | None = None,
    size: tuple[float, float] | None = None,
    axes: np.ndarray | None = None,
    par_indices: Sequence[int] = _UNSET,
    suptitle: str | None = _UNSET,
    style_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Plot marginals.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    parameter_indices: list of integer values
        List of integer values specifying which parameters to plot.
        Default: All parameters are shown.
    stepsize:
        Only one in `stepsize` values is plotted.
    plot_type: {'hist'|'kde'|'both'}
        Specify whether to plot a histogram ('hist'), a kernel density estimate
        ('kde'), or both ('both').
    bw_method: {'scott', 'silverman' | scalar | pair of scalars}
        Kernel bandwidth method.
    show_bounds:
        If ``True`` (default) draw the parameter bound lines and frame each
        panel's x-axis to include them. If ``False`` frame each panel tightly
        to its data range and omit the bound lines.
    title:
        Figure title.
    size:
        Figure size in inches.
    axes:
        Axes grid to use. Must match the computed subplot layout.
    suptitle:
        Deprecated. Use ``title`` instead.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``rectangle_color``, ``rectangle_alpha``, ``rectangle_edgecolor``,
          ``rectangle_linewidth`` — histogram bar styling.
        - ``bound_color``, ``bound_linestyle``, ``bound_linewidth``,
          ``bound_alpha`` — parameter-bound line style.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Return
    --------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.
    """
    style = resolve_style(style_kwargs)
    title = process_deprecated_kwarg("title", title, "suptitle", suptitle)

    parameter_indices = process_deprecated_kwarg(
        "parameter_indices",
        parameter_indices,
        "par_indices",
        par_indices,
    )

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub, param_names = get_data_to_plot(
        result=result,
        i_chain=i_chain,
        stepsize=stepsize,
        parameter_indices=parameter_indices,
    )

    num_row, num_col = make_grid_shape(nr_params)
    axes = get_axes_array(axes=axes, nrows=num_row, ncols=num_col, size=size)
    fig = axes.flat[0].figure
    axes = hide_unused_axes(axes=axes, n_used=nr_params, clear=True)

    par_ax = dict(zip(param_names, axes.flat))

    # Build name→index map for looking up per-parameter lb/ub/scale.
    # theta_lb/theta_ub from get_data_to_plot are the full reduced vectors,
    # but param_names may be a subset when parameter_indices is given.
    all_reduced_names = result.problem.get_reduced_vector(result.problem.x_names)
    name_to_reduced_idx = {name: i for i, name in enumerate(all_reduced_names)}
    x_scales_reduced = (
        result.problem.get_reduced_vector(result.problem.x_scales)
        if getattr(result.problem, "x_scales", None) is not None
        else None
    )

    _show_kde = plot_type in ("kde", "both")
    _show_rug = plot_type in ("hist", "both")

    for idx, par_id in enumerate(param_names):
        ax = par_ax[par_id]
        used_color = plot_density_panel(
            ax,
            np.asarray(params_fval[par_id]),
            bins="auto",
            bw_method=bw_method,
            rectangle_color=style["rectangle_color"],
            rectangle_alpha=style["rectangle_alpha"],
            rectangle_edgecolor=style["rectangle_edgecolor"],
            rectangle_linewidth=style["rectangle_linewidth"],
            show_hist=(plot_type in ("hist", "both")),
            show_kde=_show_kde,
            show_rug=_show_rug,
        )
        par_reduced_idx = name_to_reduced_idx.get(par_id, idx)
        lb_val = theta_lb[par_reduced_idx]
        ub_val = theta_ub[par_reduced_idx]

        legend_handles, legend_labels = [], []
        if used_color is not None and idx == 0:
            if _show_kde:
                legend_handles.append(Line2D([0], [0], color=used_color, lw=2))
                legend_labels.append("KDE")
            if _show_rug:
                legend_handles.append(
                    Line2D([0], [0], color=used_color, marker="|", lw=0,
                           markersize=10, markeredgewidth=1.2)
                )
                legend_labels.append("Samples")

        if not show_bounds:
            # clip x-axis to the data range instead of the parameter bounds
            vals = np.asarray(params_fval[par_id])
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size > 0:
                spread = finite_vals.max() - finite_vals.min()
                pad = max(spread * 0.05, np.abs(finite_vals).max() * 0.01)
                ax.set_xlim(finite_vals.min() - pad, finite_vals.max() + pad)
        elif np.isfinite(lb_val) and np.isfinite(ub_val):
            bound_handle = draw_bounds_1d(ax, lb_val, ub_val, axis="x", style=style)
            if idx == 0:
                legend_handles.append(bound_handle)
                legend_labels.append("Bounds")

        if legend_handles:
            ax.legend(handles=legend_handles, labels=legend_labels)

        scale = (
            x_scales_reduced[par_reduced_idx]
            if x_scales_reduced is not None
            else None
        )
        xlabel = f"{par_id} ({scale})" if scale is not None else par_id
        ax.set_xlabel(xlabel)
        # y-label only on the leftmost column to avoid grid-wide repetition
        ax.set_ylabel("Density" if idx % num_col == 0 else "")

    if title is not None:
        fig.suptitle(title)

    return axes


def get_data_to_plot(
    result: Result,
    i_chain: int,
    stepsize: int,
    full_trace: bool = False,
    parameter_indices: Sequence[int] | None = None,
):
    """Get the data which should be plotted as a pandas.DataFrame.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot.
    stepsize:
        Only one in `stepsize` values is plotted.
    full_trace:
        Keep the full length of the chain. Default: False.
    parameter_indices: list of integer values
        List of integer values specifying which parameters to plot.
        Default: All parameters are shown.

    Returns
    -------
    nr_params:
        Number of parameters to be plotted.
    params_fval:
        Log posterior values to be plotted.
    theta_lb:
        Parameter lower bounds to be plotted.
    theta_ub:
        Parameter upper bounds to be plotted.
    param_names:
        Parameter names to be plotted.
    """
    # get parameters and fval results as numpy arrays (trace_x is numpy array)
    arr_param = np.asarray(result.sample_result.trace_x[i_chain])

    if result.sample_result.burn_in is None:
        warnings.warn(
            "Burn in index not found in the results, the full chain "
            "will be shown.\nYou may want to use, e.g., "
            "`pypesto.sample.geweke_test`.",
            stacklevel=2,
        )
        _burn_in = 0
    else:
        _burn_in = result.sample_result.burn_in

    # Burn in index
    if full_trace is False:
        burn_in = _burn_in
    else:
        burn_in = 0

    sample_result: McmcPtResult = result.sample_result

    # thin out by stepsize, from the index burn_in until end of vector
    arr_param = arr_param[np.arange(burn_in, len(arr_param), stepsize)]

    # invert sign for log posterior values (trace_neglogpost is numpy array)
    arr_fval = -np.asarray(sample_result.trace_neglogpost[i_chain])
    indices = np.arange(burn_in, len(arr_fval), stepsize)
    arr_fval = arr_fval[indices]
    theta_lb = result.problem.lb
    theta_ub = result.problem.ub

    # get parameter names from all non-fixed parameters
    param_names = result.problem.get_reduced_vector(result.problem.x_names)

    # transform ndarray to pandas for the use of seaborn
    pd_params = pd.DataFrame(arr_param, columns=param_names)
    pd_fval = pd.DataFrame(data=arr_fval, columns=["logPosterior"])

    pd_iter = pd.DataFrame(data=indices, columns=["iteration"])

    if full_trace:
        converged = np.zeros(len(arr_fval))
        converged[_burn_in:] = 1
        pd_conv = pd.DataFrame(data=converged, columns=["converged"])

        params_fval = pd.concat(
            [pd_params, pd_fval, pd_iter, pd_conv], axis=1, ignore_index=False
        )
    else:
        params_fval = pd.concat(
            [pd_params, pd_fval, pd_iter], axis=1, ignore_index=False
        )

    # some global parameters
    nr_params = arr_param.shape[1]  # number of parameters

    if parameter_indices is not None:
        param_names = params_fval.columns.values[parameter_indices]
        nr_params = len(parameter_indices)
    else:
        param_names = params_fval.columns.values[0:nr_params]

    return nr_params, params_fval, theta_lb, theta_ub, param_names
