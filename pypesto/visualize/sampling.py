import logging
import warnings
from collections.abc import Sequence
from colorsys import rgb_to_hls
from typing import Optional, Union

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
from .misc import rgba2rgb

cmap = matplotlib.cm.viridis
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
    title: str = None,
    size: tuple[float, float] = None,
    ax: matplotlib.axes.Axes = None,
):
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

    # set axes and figure
    if ax is None:
        _, ax = plt.subplots(figsize=size)

    sns.set(style="ticks")
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
    levels: Union[float, Sequence[float]],
    labels: dict[str, str],
    level_opacities: Sequence[float],
    variable_names: Sequence[str],
    variable_colors: Sequence[RGB],
    groupby: str,
    artist_padding: float,
    n_col: int,
    average: str,
    add_sd: bool,
    grouped_measurements: Optional[
        dict[tuple[str, str], Sequence[Sequence[float]]]
    ],
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
    levels: Union[float, Sequence[float]],
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
        list(cmap(v))[:LEN_RGB]
        for v in np.linspace(cmap_min, cmap_max, n_variables)
    ]

    return level_opacities, variable_colors


def sampling_prediction_trajectories(
    ensemble_prediction: EnsemblePrediction,
    levels: Union[float, Sequence[float]],
    title: str = None,
    size: tuple[float, float] = None,
    axes: matplotlib.axes.Axes = None,
    labels: dict[str, str] = None,
    axis_label_padding: int = 50,
    groupby: str = CONDITION,
    condition_gap: float = 0.01,
    condition_ids: Sequence[str] = None,
    output_ids: Sequence[str] = None,
    weighting: bool = False,
    reverse_opacities: bool = False,
    average: str = MEDIAN,
    add_sd: bool = False,
    measurement_df: pd.DataFrame = None,
) -> matplotlib.axes.Axes:
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
        The plot axes.
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

    if axes is None:
        n_row = int(np.round(np.sqrt(n_subplots)))
        n_col = int(np.ceil(n_subplots / n_row))
        fig, axes = plt.subplots(n_row, n_col, figsize=size, squeeze=False)
        for ax in axes.flat[n_subplots:]:
            ax.remove()
    else:
        fig = axes.get_figure()
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        if len(axes.flat) < n_subplots:
            raise ValueError(
                "Provided `axes` contains insufficient subplots. At least "
                f"{n_subplots} are required."
            )
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
    xmin = min(ax.get_position().xmin for ax in axes.flat)
    ymin = min(ax.get_position().ymin for ax in axes.flat)
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
    alpha: Sequence[int] = None,
    step: float = 0.05,
    show_median: bool = True,
    title: str = None,
    size: tuple[float, float] = None,
    ax: matplotlib.axes.Axes = None,
) -> matplotlib.axes.Axes:
    """
    Plot MCMC-based parameter credibility intervals.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    alpha:
        List of lower tail probabilities, defaults to 95% interval.
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
    if alpha is None:
        alpha = [95]

    # automatically sort values in decreasing order
    alpha_sorted = sorted(alpha, reverse=True)
    # define colormap
    evenly_spaced_interval = np.linspace(0, 1, len(alpha_sorted))
    colors = [plt.cm.tab20c_r(x) for x in evenly_spaced_interval]
    # number of sampled parameters
    n_pars = result.sample_result.trace_x.shape[-1]

    # set axes and figure
    if ax is None:
        _, ax = plt.subplots(figsize=size)

    # loop over parameters
    for npar in range(n_pars):
        # initialize height of boxes
        _step = step
        # loop over confidence levels
        for n, level in enumerate(alpha_sorted):
            # extract percentile-based confidence intervals
            lb, ub = calculate_ci_mcmc_sample(
                result=result,
                ci_level=level / 100,
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
                label=str(level) + "% CI",
            )

            if show_median:
                if n == len(alpha_sorted) - 1:
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
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1))

    return ax


def sampling_parameter_traces(
    result: Result,
    i_chain: int = 0,
    par_indices: Sequence[int] = None,
    full_trace: bool = False,
    stepsize: int = 1,
    use_problem_bounds: bool = True,
    suptitle: str = None,
    size: tuple[float, float] = None,
    ax: matplotlib.axes.Axes = None,
):
    """
    Plot parameter values over iterations.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    par_indices: list of integer values
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
    ax:
        Axes object to use.

    Returns
    -------
    ax:
        The plot axes.
    """
    import seaborn as sns

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub, param_names = get_data_to_plot(
        result=result,
        i_chain=i_chain,
        stepsize=stepsize,
        full_trace=full_trace,
        par_indices=par_indices,
    )

    # compute, how many rows and columns we need for the subplots
    num_row = int(np.round(np.sqrt(nr_params)))
    num_col = int(np.ceil(nr_params / num_row))

    # set axes and figure
    if ax is None:
        fig, ax = plt.subplots(num_row, num_col, squeeze=False, figsize=size)
    else:
        fig = ax.get_figure()

    par_ax = dict(zip(param_names, ax.flat))

    sns.set(style="ticks")
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

    fig.tight_layout()
    sns.despine()

    return ax


def sampling_scatter(
    result: Result,
    i_chain: int = 0,
    stepsize: int = 1,
    suptitle: str = None,
    diag_kind: str = "kde",
    size: tuple[float, float] = None,
    show_bounds: bool = True,
):
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
    ax:
        The plot axes.
    """
    import seaborn as sns

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub, _ = get_data_to_plot(
        result=result, i_chain=i_chain, stepsize=stepsize
    )

    sns.set(style="ticks")

    # TODO: Think this throws the axis errors in seaborn.
    ax = sns.pairplot(
        params_fval.drop(["logPosterior", "iteration"], axis=1),
        diag_kind=diag_kind,
    )

    if size is not None:
        ax.fig.set_size_inches(size)

    if suptitle:
        ax.fig.suptitle(suptitle)

    if show_bounds:
        # set bounds of plot to parameter bounds. Only use diagonal as
        # sns.PairGrid has sharex,sharey = True by default.
        for i_axis, axis in enumerate(np.diag(ax.axes)):
            axis.set_xlim(result.problem.lb[i_axis], result.problem.ub[i_axis])
            axis.set_ylim(result.problem.lb[i_axis], result.problem.ub[i_axis])

    return ax


def sampling_1d_marginals(
    result: Result,
    i_chain: int = 0,
    par_indices: Sequence[int] = None,
    stepsize: int = 1,
    plot_type: str = "both",
    bw_method: str = "scott",
    suptitle: str = None,
    size: tuple[float, float] = None,
):
    """
    Plot marginals.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    par_indices: list of integer values
        List of integer values specifying which parameters to plot.
        Default: All parameters are shown.
    stepsize:
        Only one in `stepsize` values is plotted.
    plot_type: {'hist'|'kde'|'both'}
        Specify whether to plot a histogram ('hist'), a kernel density estimate
        ('kde'), or both ('both').
    bw_method: {'scott', 'silverman' | scalar | pair of scalars}
        Kernel bandwidth method.
    suptitle:
        Figure super title.
    size:
        Figure size in inches.

    Return
    --------
    ax:
        matplotlib-axes
    """
    import seaborn as sns

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub, param_names = get_data_to_plot(
        result=result,
        i_chain=i_chain,
        stepsize=stepsize,
        par_indices=par_indices,
    )

    # compute, how many rows and columns we need for the subplots
    num_row = int(np.round(np.sqrt(nr_params)))
    num_col = int(np.ceil(nr_params / num_row))

    fig, ax = plt.subplots(num_row, num_col, squeeze=False, figsize=size)

    par_ax = dict(zip(param_names, ax.flat))
    sns.set(style="ticks")

    # fig, ax = plt.subplots(nr_params, figsize=size)[1]
    for idx, par_id in enumerate(param_names):
        if plot_type == "kde":
            # TODO: add bw_adjust as option?
            sns.kdeplot(
                params_fval[par_id], bw_method=bw_method, ax=par_ax[par_id]
            )
        elif plot_type == "hist":
            # fixes usage of sns distplot which throws a future warning
            sns.histplot(
                x=params_fval[par_id], ax=par_ax[par_id], stat="density"
            )
            sns.rugplot(x=params_fval[par_id], ax=par_ax[par_id])
        elif plot_type == "both":
            sns.histplot(
                x=params_fval[par_id],
                kde=True,
                ax=par_ax[par_id],
                stat="density",
            )
            sns.rugplot(x=params_fval[par_id], ax=par_ax[par_id])

        par_ax[par_id].set_xlabel(param_names[idx])
        par_ax[par_id].set_ylabel("Density")

    sns.despine()

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()

    return ax


def get_data_to_plot(
    result: Result,
    i_chain: int,
    stepsize: int,
    full_trace: bool = False,
    par_indices: Sequence[int] = None,
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
    par_indices: list of integer values
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

    if par_indices is not None:
        param_names = params_fval.columns.values[par_indices]
        nr_params = len(par_indices)
    else:
        param_names = params_fval.columns.values[0:nr_params]

    return nr_params, params_fval, theta_lb, theta_ub, param_names
