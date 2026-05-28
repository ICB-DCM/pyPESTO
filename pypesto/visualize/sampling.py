import logging
import warnings
from collections.abc import Sequence
from colorsys import rgb_to_hls

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D

from ..C import (
    CONDITION,
    LEN_RGB,
    MEDIAN,
    OUTPUT,
    RGB,
    RGBA_BLACK,
    RGBA_MAX,
    RGBA_MIN,
    STANDARD_DEVIATION,
)
from ..ensemble import EnsemblePrediction, get_percentile_label
from ..result import McmcPtResult, PredictionResult, Result
from ..sample import calculate_ci_mcmc_sample
from ._style import resolve_style
from .misc import (
    _UNSET,
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


prediction_errorbar_settings = {
    "fmt": "none",
    "color": "k",
    "capsize": 10,
}


def sampling_fval_traces(
    result: Result,
    i_chain: int = 0,
    full_trace: bool = False,
    stepsize: int = 1,
    title: str | None = None,
    size: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
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
        Axes title.
    size: ndarray
        Figure size in inches.
    ax:
        Axes object to use.

    Returns
    -------
    ax:
        The plot axes.
    """
    import seaborn as sns

    # get data which should be plotted
    _, params_fval, _, _, _ = get_data_to_plot(
        result=result,
        i_chain=i_chain,
        stepsize=stepsize,
        full_trace=full_trace,
    )

    ax = get_ax(ax, size)

    kwargs = {"edgecolor": "w", "linewidth": 0.3, "s": 10}  # for edge color
    if full_trace:
        kwargs["hue"] = "converged"
        if len(params_fval[kwargs["hue"]].unique()) == 1:
            kwargs["palette"] = ["#477ccd"]
        elif len(params_fval[kwargs["hue"]].unique()) == 2:
            kwargs["palette"] = ["#868686", "#477ccd"]
        kwargs["legend"] = False

    sns.scatterplot(
        x="iteration", y="logPosterior", data=params_fval, ax=ax, **kwargs
    )

    if result.sample_result.burn_in is None:
        _burn_in = 0
    else:
        _burn_in = result.sample_result.burn_in

    if full_trace and _burn_in > 0:
        ax.axvline(_burn_in, linestyle="--", linewidth=1.5, color="k")

    ax.set_xlabel("iteration index")
    ax.set_ylabel("log-posterior")

    if title:
        ax.set_title(title)

    sns.despine()

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
    level_opacities: dict[int, float],
    labels: dict[str, str],
    variable_colors: Sequence[RGB],
    average: str = MEDIAN,
    add_sd: bool = False,
    grouped_measurements: dict[
        tuple[str, str], Sequence[Sequence[float]]
    ] = None,
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
        The axes to plot with. Should contain atleast `len(output_ids)`
        subplots.
    levels:
        Credibility levels, e.g. [95] for a 95% credibility interval. See the
        :py:func:`_get_level_percentiles` method for a description of how these
        levels are handled, and current limitations.
    level_opacities:
        A mapping from the credibility levels to the opacities that they should
        be plotted with. Opacity is the only thing that differentiates
        credibility levels in the resulting plot.
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
    # Each subplot has all data for a single condition.
    for condition_index, condition_id in enumerate(condition_ids):
        ax = axes.flat[condition_index]
        ax.set_title(f"Condition: {labels[condition_id]}")
        # Each subplot has all data for all condition-specific outputs.
        for output_index, output_id in enumerate(output_ids):
            facecolor0 = variable_colors[output_index]
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
                "k-",
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
                    **prediction_errorbar_settings,
                )
            # Plot the regions described by the credibility level,
            # for each output.
            for level_index, level in enumerate(levels):
                # Get the percentiles that correspond to the credibility level,
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
                    marker="o",
                    facecolor=facecolor0,
                    edgecolor=(
                        "white"
                        if rgb_to_hls(*facecolor0)[1] < 0.5
                        else "black"
                    ),
                )


def _plot_trajectories_by_output(
    summary: dict[str, PredictionResult],
    condition_ids: Sequence[str],
    output_ids: Sequence[str],
    axes: matplotlib.axes.Axes,
    levels: Sequence[float],
    level_opacities: dict[int, float],
    labels: dict[str, str],
    variable_colors: Sequence[RGB],
    average: str = MEDIAN,
    add_sd: bool = False,
    grouped_measurements: dict[
        tuple[str, str], Sequence[Sequence[float]]
    ] = None,
) -> None:
    """Plot predicted trajectories, with subplots grouped by output.

    Each subplot is further divided by conditions, such that all conditions
    are displayed side-by-side for a single output. Hence, in each subplot, the
    timepoints of each condition plot are shifted by the the end timepoint of
    the previous condition plot. For examples of this, see the plots with
    `groupby=OUTPUT` in the example notebook
    `doc/example/sampling_diagnostics.ipynb`.

    See :py:func:`_plot_trajectories_by_condition` for parameter descriptions.
    """
    # Each subplot has all data for a single output.
    for output_index, output_id in enumerate(output_ids):
        # Store the end timepoint of the previous condition plot, such that the
        # next condition plot starts at the end of the previous condition plot.
        t0 = 0
        ax = axes.flat[output_index]
        ax.set_title(f"Trajectory: {labels[output_id]}")
        # Each subplot is divided by conditions, with vertical lines.
        for condition_index, condition_id in enumerate(condition_ids):
            facecolor0 = variable_colors[condition_index]
            if condition_index != 0:
                ax.axvline(
                    t0,
                    linewidth=2,
                    color="k",
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
                "k-",
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
                    **prediction_errorbar_settings,
                )
            t_max = max(t_max, *t_average_shifted)
            for level_index, level in enumerate(levels):
                # Get the percentiles that correspond to the credibility level,
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
                    marker="o",
                    facecolor=facecolor0,
                    edgecolor=(
                        "white"
                        if rgb_to_hls(*facecolor0)[1] < 0.5
                        else "black"
                    ),
                )
            # Set t0 to the last plotted timepoint of the current condition
            # plot.
            t0 = t_max


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
    fig: matplotlib.figure.Figure,
    axes: matplotlib.axes.Axes,
    levels: float | Sequence[float],
    labels: dict[str, str],
    level_opacities: Sequence[float],
    variable_names: Sequence[str],
    variable_colors: Sequence[RGB],
    groupby: str,
    artist_padding: float,
    n_col: int,
    average: str,
    add_sd: bool,
    grouped_measurements: dict[tuple[str, str], Sequence[Sequence[float]]]
    | None,
) -> None:
    """Add legends to a sampling prediction trajectories plot.

    Create a dummy plot from fake data such that it can be used to produce
    appropriate legends.

    Variable here refers to the thing that differs in the plot. For example, if
    the call to :py:func:`sampling_prediction_trajectories` has
    `groupby=OUTPUT`, then the variable is `CONDITION`. Similarly, if
    `groupby=CONDITION`, then the variable is `OUTPUT`.

    Parameters
    ----------
    fig:
        The figure to add the legends to.
    axes:
        The axes of the figure to add the legend to.
    levels:
        The credibility levels.
    labels:
        The labels for the IDs in the plot.
    level_opacities:
        The opacity to plot each credibility level with.
    variable_names:
        The name of each variable.
    variable_colors:
        The color to plot each variable in.
    groupby:
        The grouping of data in the subplots.
    artist_padding:
        The padding between the figure and the legends.
    n_col:
        The number of columns of subplots in the figure.
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
    # Fake plots for legend line styles
    fake_data = [[0], [0]]
    variable_lines = np.array(
        [
            # Assumes that the color for a variable is always the same, with
            # different opacity for different credibility interval levels.
            # Create a line object with fake data for each variable value.
            [
                labels[variable_name],
                Line2D(*fake_data, color=variable_colors[index], lw=4),
            ]
            for index, variable_name in enumerate(variable_names)
        ]
    )
    # Assumes that different CI levels are represented as
    # different opacities of the same color.
    # Create a line object with fake data for each credibility level.
    ci_lines = []
    for index, level in enumerate(levels):
        ci_lines.append(
            [
                f"{level}% CI",
                Line2D(
                    *fake_data,
                    color=rgba2rgb(
                        [*RGBA_BLACK[:LEN_RGB], level_opacities[index]]
                    ),
                    lw=4,
                ),
            ]
        )

    # Create a line object with fake data for the average line.
    average_title = average.title()
    average_line_object_line2d = Line2D(*fake_data, color=RGBA_BLACK)
    if add_sd:
        capline = Line2D(
            *fake_data,
            color=prediction_errorbar_settings["color"],
            # https://github.com/matplotlib/matplotlib/blob
            # /710fce3df95e22701bd68bf6af2c8adbc9d67a79/lib/matplotlib/
            # axes/_axes.py#L3424=
            markersize=2.0 * prediction_errorbar_settings["capsize"],
        )
        average_title += " + SD"
        barline = LineCollection(
            np.empty((2, 2, 2)),
            color=prediction_errorbar_settings["color"],
        )
        average_line_object = ErrorbarContainer(
            (
                average_line_object_line2d,
                [capline],
                [barline],
            ),
            has_yerr=True,
        )
    else:
        average_line_object = average_line_object_line2d
    average_line = [[average_title, average_line_object]]

    # Create a line object with fake data for the data points.
    data_line = []
    if grouped_measurements:
        data_line = [
            [
                "Data",
                Line2D(
                    *fake_data,
                    linewidth=0,
                    marker="o",
                    markerfacecolor="grey",
                    markeredgecolor="white",
                ),
            ]
        ]

    level_lines = np.array(ci_lines + average_line + data_line)

    # CI level, and variable name, legends.
    legend_options_top_right = {
        "bbox_to_anchor": (1 + artist_padding, 1),
        "loc": "upper left",
    }
    legend_options_bottom_right = {
        "bbox_to_anchor": (1 + artist_padding, 0),
        "loc": "lower left",
    }
    legend_titles = {
        OUTPUT: "Conditions",
        CONDITION: "Trajectories",
    }
    legend_variables = axes.flat[n_col - 1].legend(
        variable_lines[:, 1],
        variable_lines[:, 0],
        **legend_options_top_right,
        title=legend_titles[groupby],
    )
    # Legend for CI levels
    axes.flat[-1].legend(
        level_lines[:, 1],
        level_lines[:, 0],
        **legend_options_bottom_right,
        title="Prediction",
    )
    fig.add_artist(legend_variables)


def _handle_colors(
    levels: float | Sequence[float],
    n_variables: int,
    reverse: bool = False,
) -> tuple[Sequence[float], Sequence[RGB]]:
    """Calculate the colors for the prediction trajectories plot.

    Parameters
    ----------
    levels:
        The credibility levels.
    n_variables:
        The maximum possible number of variables per subplot.

    Returns
    -------
    A 2-tuple, with the following indices and values.
    - `0`: a list of opacities, one per level.
    - `1`: a list of colors, one per variable.
    """
    level_opacities = sorted(
        # min 30%, max 100%, opacity
        np.linspace(0.3 * RGBA_MAX, RGBA_MAX, len(levels)),
        reverse=reverse,
    )
    cmap_min = RGBA_MIN
    cmap_max = 0.85 * (RGBA_MAX - RGBA_MIN) + RGBA_MIN  # exclude yellows

    # define colormap
    variable_colors = [
        list(matplotlib.cm.viridis(v))[:LEN_RGB]
        for v in np.linspace(cmap_min, cmap_max, n_variables)
    ]

    return level_opacities, variable_colors


def sampling_prediction_trajectories(
    ensemble_prediction: EnsemblePrediction,
    levels: float | Sequence[float],
    title: str | None = None,
    size: tuple[float, float] | None = None,
    axes: matplotlib.axes.Axes | np.ndarray | None = None,
    labels: dict[str, str] | None = None,
    axis_label_padding: int = 50,
    groupby: str = CONDITION,
    condition_gap: float = 0.01,
    condition_ids: Sequence[str] = None,
    output_ids: Sequence[str] = None,
    weighting: bool = False,
    reverse_opacities: bool = False,
    average: str = MEDIAN,
    add_sd: bool = False,
    measurement_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Visualize prediction trajectory of an EnsemblePrediction.

    Plot MCMC-based prediction credibility intervals for the
    model states or outputs. One or various credibility levels
    can be depicted. Plots are grouped by condition.

    Parameters
    ----------
    ensemble_prediction:
        The ensemble prediction.
    levels:
        Credibility levels, e.g. [95] for a 95% credibility interval. See the
        :py:func:`_get_level_percentiles` method for a description of how these
        levels are handled, and current limitations.
    title:
        Axes title.
    size: ndarray
        Figure size in inches.
    axes:
        Axes object to use.
    labels:
        Keys should be ensemble output IDs, values should be the desired
        label for that output. Defaults to output IDs.
    axis_label_padding:
        Pixels between axis labels and plots.
    groupby:
        Group plots by `pypesto.C.OUTPUT` or
        `pypesto.C.CONDITION`.
    condition_gap:
        Gap between conditions when
        `groupby == pypesto.C.CONDITION`.
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
    measurement_df:
        Plot measurement data. NB: This should take the form of a PEtab
        measurements table, and the `observableId` column should correspond
        to the output IDs in the ensemble prediction.

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.
    """
    if labels is None:
        labels = {}
    if len(list(levels)) == 1:
        levels = list(levels)
    levels = sorted(levels, reverse=True)
    # Get the percentiles that correspond to the requested credibility levels.
    percentiles = [
        percentile
        for level in levels
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

    # Handle data
    grouped_measurements = {}
    if measurement_df is not None:
        import petab.v1 as petab

        for condition_id in condition_ids:
            if petab.PARAMETER_SEPARATOR in condition_id:
                (
                    preequilibration_condition_id,
                    simulation_condition_id,
                ) = condition_id.split(petab.PARAMETER_SEPARATOR)
            else:
                preequilibration_condition_id, simulation_condition_id = (
                    "",
                    condition_id,
                )
            condition = {
                petab.SIMULATION_CONDITION_ID: simulation_condition_id,
            }
            if preequilibration_condition_id:
                condition[petab.PREEQUILIBRATION_CONDITION_ID] = (
                    preequilibration_condition_id
                )
            for output_id in output_ids:
                _df = petab.get_rows_for_condition(
                    measurement_df=measurement_df,
                    condition=condition,
                )
                _df = _df.loc[_df[petab.OBSERVABLE_ID] == output_id]
                grouped_measurements[(condition_id, output_id)] = [
                    _df[petab.TIME],
                    _df[petab.MEASUREMENT],
                ]

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
        levels=levels,
        n_variables=n_variables,
        reverse=reverse_opacities,
    )

    n_row = int(np.round(np.sqrt(n_subplots)))
    n_col = int(np.ceil(n_subplots / n_row))

    axes = get_axes_array(axes=axes, nrows=n_row, ncols=n_col, size=size)
    fig = axes.flat[0].figure
    axes = hide_unused_axes(axes=axes, n_used=n_subplots, clear=True)
    artist_padding = axis_label_padding / (fig.get_size_inches() * fig.dpi)[0]

    if groupby == CONDITION:
        _plot_trajectories_by_condition(
            summary=summary,
            condition_ids=condition_ids,
            output_ids=output_ids,
            axes=axes,
            levels=levels,
            level_opacities=level_opacities,
            labels=labels,
            variable_colors=variable_colors,
            average=average,
            add_sd=add_sd,
            grouped_measurements=grouped_measurements,
        )
    elif groupby == OUTPUT:
        _plot_trajectories_by_output(
            summary=summary,
            condition_ids=condition_ids,
            output_ids=output_ids,
            axes=axes,
            levels=levels,
            level_opacities=level_opacities,
            labels=labels,
            variable_colors=variable_colors,
            average=average,
            add_sd=add_sd,
            grouped_measurements=grouped_measurements,
        )

    if title:
        fig.suptitle(title)

    _handle_legends(
        fig=fig,
        axes=axes,
        levels=levels,
        labels=labels,
        level_opacities=level_opacities,
        variable_names=variable_names,
        variable_colors=variable_colors,
        groupby=groupby,
        artist_padding=artist_padding,
        n_col=n_col,
        average=average,
        add_sd=add_sd,
        grouped_measurements=grouped_measurements,
    )

    # X and Y labels
    visible_axes = [ax for ax in axes.flat if ax.get_visible()]
    xmin = min(ax.get_position().xmin for ax in visible_axes)
    ymin = min(ax.get_position().ymin for ax in visible_axes)
    xlabel = (
        "Cumulative time across all conditions"
        if groupby == OUTPUT
        else "Time"
    )
    fig.text(
        0.5,
        ymin - artist_padding,
        xlabel,
        ha="center",
        va="center",
        transform=fig.transFigure,
    )
    fig.text(
        xmin - artist_padding,
        0.5,
        "Simulated values",
        ha="center",
        va="center",
        transform=fig.transFigure,
        rotation="vertical",
    )

    # plt.tight_layout()  # Ruins layout for `groupby == OUTPUT`.
    return axes


def sampling_parameter_cis(
    result: Result,
    confidence_levels: Sequence[float] = None,
    step: float = 0.05,
    show_median: bool = True,
    title: str | None = None,
    size: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    alpha: Sequence[int] = None,
) -> matplotlib.axes.Axes:
    """
    Plot MCMC-based parameter credibility intervals.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    confidence_levels:
        Credibility levels as fractions in (0, 1), e.g. ``[0.95]`` for a
        95% credibility interval. Defaults to ``[0.95]``.
    alpha:
        Deprecated. Use ``confidence_levels`` instead.
        Previously accepted integer percentages (e.g. ``[95]``); values
        are divided by 100 automatically during the transition.
    step:
        Height of boxes for projectile plot, defaults to 0.05.
    show_median:
        Plot the median of the MCMC chain. Default: True.
    title:
        Axes title.
    size: ndarray
        Figure size in inches.
    ax:
        Axes object to use.

    Returns
    -------
    ax:
        The plot axes.
    """
    if alpha is not None:
        if confidence_levels is not None:
            raise ValueError(
                "Pass either `confidence_levels` or the deprecated `alpha`, not both."
            )
        import warnings

        warnings.warn(
            "`alpha` is deprecated; use `confidence_levels` instead. "
            "Note: units have changed — pass fractions in (0, 1) "
            "(e.g. `confidence_levels=[0.95]`) instead of integer percentages "
            "(e.g. `alpha=[95]`). Your values have been divided by 100 automatically.",
            DeprecationWarning,
            stacklevel=2,
        )
        confidence_levels = [a / 100 for a in alpha]

    if confidence_levels is None:
        confidence_levels = [0.95]

    # automatically sort values in decreasing order
    levels_sorted = sorted(confidence_levels, reverse=True)
    # define colormap
    evenly_spaced_interval = np.linspace(0, 1, len(levels_sorted))
    colors = [plt.cm.tab20c_r(x) for x in evenly_spaced_interval]
    # number of sampled parameters
    n_pars = result.sample_result.trace_x.shape[-1]

    ax = get_ax(ax, size)

    # loop over parameters
    for npar in range(n_pars):
        # initialize height of boxes
        _step = step
        # loop over confidence levels
        for n, level in enumerate(levels_sorted):
            # extract percentile-based confidence intervals
            lb, ub = calculate_ci_mcmc_sample(
                result=result,
                ci_level=level,
            )

            # assemble boxes for projectile plot
            x1 = [lb[npar], ub[npar]]
            y1 = [npar + _step, npar + _step]
            y2 = [npar - _step, npar - _step]
            # Plot boxes
            ax.fill(
                np.append(x1, x1[::-1]),
                np.append(y1, y2[::-1]),
                color=colors[n],
                label=f"{level:.0%} CI",
            )

            if show_median:
                if n == len(levels_sorted) - 1:
                    burn_in = result.sample_result.burn_in
                    converged = result.sample_result.trace_x[0, burn_in:, npar]
                    _median = np.median(converged)
                    ax.plot(
                        [_median, _median],
                        [npar - _step, npar + _step],
                        "k-",
                        label="MCMC median",
                    )

            # increment height of boxes
            _step += step

    ax.set_yticks(range(n_pars))
    ax.set_yticklabels(
        result.problem.get_reduced_vector(result.problem.x_names)
    )
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Parameter name")

    if title:
        ax.set_title(title)

    # handle legend
    plt.gca().invert_yaxis()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1))

    return ax


def sampling_parameter_traces(
    result: Result,
    i_chain: int = 0,
    parameter_indices: Sequence[int] = None,
    full_trace: bool = False,
    stepsize: int = 1,
    use_problem_bounds: bool = True,
    suptitle: str | None = None,
    size: tuple[float, float] | None = None,
    axes: np.ndarray | None = None,
    ax: np.ndarray | None = _UNSET,
    par_indices: Sequence[int] = _UNSET,
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
        Defines if the y-limits shall be the lower and upper bounds of
        parameter estimation problem.
    suptitle:
        Figure suptitle.
    size:
        Figure size in inches.
    axes:
        Axes grid to use. Must match the computed subplot layout.
    ax:
        Deprecated. Use ``axes`` instead.
    par_indices:
        Deprecated. Use ``parameter_indices`` instead.

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.
    """
    parameter_indices = process_deprecated_kwarg(
        "parameter_indices",
        parameter_indices,
        "par_indices",
        par_indices,
    )
    axes = process_deprecated_kwarg("axes", axes, "ax", ax)

    import seaborn as sns

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub, param_names = get_data_to_plot(
        result=result,
        i_chain=i_chain,
        stepsize=stepsize,
        full_trace=full_trace,
        parameter_indices=parameter_indices,
    )

    num_row, num_col = make_grid_shape(nr_params)
    if size is None and axes is None:
        size = (3.5 * num_col, 2.5 * num_row)
    axes = get_axes_array(axes=axes, nrows=num_row, ncols=num_col, size=size)
    fig = axes.flat[0].figure
    axes = hide_unused_axes(axes=axes, n_used=nr_params, clear=True)

    par_ax = dict(zip(param_names, axes.flat, strict=True))

    kwargs = {"edgecolor": "w", "linewidth": 0.3, "s": 10}  # for edge color

    if full_trace:
        kwargs["hue"] = "converged"
        if len(params_fval[kwargs["hue"]].unique()) == 1:
            kwargs["palette"] = ["#477ccd"]
        elif len(params_fval[kwargs["hue"]].unique()) == 2:
            kwargs["palette"] = ["#868686", "#477ccd"]
        kwargs["legend"] = False

    if result.sample_result.burn_in is None:
        _burn_in = 0
    else:
        _burn_in = result.sample_result.burn_in

    for idx, plot_id in enumerate(param_names):
        _ax = par_ax[plot_id]

        _ax = sns.scatterplot(
            x="iteration",
            y=plot_id,
            data=params_fval,
            ax=_ax,
            **kwargs,
        )

        if full_trace and _burn_in > 0:
            _ax.axvline(
                _burn_in,
                linestyle="--",
                linewidth=1.5,
                color="k",
            )

        _ax.set_xlabel("iteration index")
        _ax.set_ylabel(param_names[idx])
        if use_problem_bounds:
            _ax.set_ylim([theta_lb[idx], theta_ub[idx]])

    if suptitle:
        fig.suptitle(suptitle)
    sns.despine()

    return axes


def sampling_scatter(
    result: Result,
    i_chain: int = 0,
    stepsize: int = 1,
    suptitle: str | None = None,
    diag_kind: str = "kde",
    size: tuple[float, float] | None = None,
    show_bounds: bool = True,
    axes: np.ndarray | None = None,
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
    suptitle:
        Figure super title.
    diag_kind:
        Visualization mode for marginal densities {‘auto’, ‘hist’, ‘kde’, None}
    size:
        Figure size in inches.
    show_bounds:
        Whether to show, and extend the plot to, the lower and upper bounds.

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.
    """
    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub, param_names = get_data_to_plot(
        result=result, i_chain=i_chain, stepsize=stepsize
    )

    if size is None and axes is None:
        size = (2.5 * nr_params + 0.5, 2.5 * nr_params + 0.5)

    axes = get_axes_array(
        axes=axes, nrows=nr_params, ncols=nr_params, size=size
    )
    fig = axes.flat[0].figure
    for ax in axes.flat:
        ax.clear()
        ax.set_visible(True)

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
                    ax=ax, values=col_vals, diag_kind=diag_kind
                )
            else:
                ax.scatter(
                    col_vals,
                    row_vals,
                    color="C0",
                    alpha=0.85,
                    s=35,
                    linewidths=0.6,
                    edgecolors="white",
                    zorder=3,
                )
                ax.set_ylabel(row_name)

            ax.set_xlabel(col_name)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    if show_bounds:
        for col in range(nr_params):
            xlim = (theta_lb[col], theta_ub[col])
            for row in range(nr_params):
                axes[row, col].set_xlim(xlim)
        for row in range(nr_params):
            ylim = (theta_lb[row], theta_ub[row])
            for col in range(nr_params):
                if row != col:
                    axes[row, col].set_ylim(ylim)

    if suptitle:
        fig.suptitle(suptitle)

    return axes


def sampling_1d_marginals(
    result: Result,
    i_chain: int = 0,
    parameter_indices: Sequence[int] | None = None,
    stepsize: int = 1,
    plot_type: str = "both",
    bins: int | str = "auto",
    bw_method: str = "scott",
    show_bounds: bool = True,
    title: str | None = None,
    size: tuple[float, float] | None = None,
    axes: np.ndarray | None = None,
    style_kwargs: dict | None = None,
    par_indices: Sequence[int] = _UNSET,
    suptitle: str | None = _UNSET,
) -> np.ndarray:
    """
    Plot 1-D marginals of the sampled parameters as histogram + KDE + rug.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: first chain.
    parameter_indices:
        Which parameters to plot, as a list of indices. Default: all parameters.
    stepsize:
        Thinning factor — plot every ``stepsize``-th sample (``1`` = all).
        Reduces overplotting and speeds up rendering for long chains.
    plot_type: {'hist'|'kde'|'both'}
        Histogram only, KDE line only, or both with rug marks (default).
    bins:
        Number of bins, or a matplotlib binning strategy (``'auto'``,
        ``'sturges'``, …). Passed to ``ax.hist``.
    bw_method: {'scott', 'silverman' | scalar | pair of scalars}
        Kernel bandwidth method for the KDE overlay.
    show_bounds:
        If ``True`` (default) draw the parameter bound lines and frame each
        panel's x-axis to include them; if ``False`` frame each panel tightly
        to its data.
    title:
        Figure title. Default: none (grids omit a title by default).
    size:
        Figure size in inches. When ``None`` the grid uses
        ``GRID_SIZE_PER_COL * num_col`` × ``GRID_SIZE_PER_ROW * num_row``
        (defaults from :mod:`pypesto.visualize._style`).
    axes:
        Axes grid to use. Must match the computed subplot layout.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``rectangle_color``, ``rectangle_alpha``, ``rectangle_edgecolor``,
          ``rectangle_linewidth`` — histogram bar styling.
        - ``line_color``, ``linewidth`` — KDE curve styling.
        - ``dash_color``, ``dash_linewidth``, ``dash_markersize``,
          ``dash_alpha`` — rug-mark styling.
        - ``bound_color``, ``bound_linestyle``, ``bound_linewidth``,
          ``bound_alpha`` — parameter-bound line styling.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.
    par_indices:
        Deprecated. Use ``parameter_indices`` instead.
    suptitle:
        Deprecated. Use ``title`` instead.

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

    par_ax = dict(zip(param_names, axes.flat[:nr_params], strict=True))

    # Build name→index map for looking up per-parameter lb/ub/scale.
    all_reduced_names = result.problem.get_reduced_vector(
        result.problem.x_names
    )
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
        vals = np.asarray(params_fval[par_id])
        finite_vals = vals[np.isfinite(vals)]
        par_reduced_idx = name_to_reduced_idx.get(par_id, idx)
        lb_val = theta_lb[par_reduced_idx]
        ub_val = theta_ub[par_reduced_idx]

        bound_handle = plot_density_panel(
            ax,
            vals,
            bins=bins,
            bw_method=bw_method,
            style=style,
            show_hist=(plot_type in ("hist", "both")),
            show_kde=_show_kde,
            show_rug=_show_rug,
            show_bounds=show_bounds,
            lb=lb_val,
            ub=ub_val,
        )

        legend_handles, legend_labels = [], []
        if finite_vals.size > 0 and idx == 0:
            if _show_kde:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=style["line_color"],
                        lw=style["linewidth"],
                    )
                )
                legend_labels.append("KDE")
            if _show_rug:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=style["dash_color"],
                        marker="|",
                        lw=0,
                        markersize=style["dash_markersize"],
                        markeredgewidth=style["dash_linewidth"],
                    )
                )
                legend_labels.append("Samples")

        if bound_handle is not None and idx == 0:
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
    parameter_indices: Sequence[int] = None,
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
