from collections.abc import Sequence
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import is_color_like
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ..C import COLOR
from ..problem import Problem
from ..result import Result
from .clust_color import assign_colors
from .misc import process_result_list
from .reference_points import ReferencePoint, create_references


def _parameter_label(problem: Problem, idx: int) -> str:
    """Return a scale-aware axis label for parameter ``idx``."""
    name = problem.x_names[idx]
    scale = problem.x_scales[idx] if problem.x_scales is not None else "lin"
    if scale == "log10":
        return f"log10({name})"
    if scale == "log":
        return f"log({name})"
    return name


# Fraction of the bound range added to axis limits so bound lines (drawn
# at the true lb/ub) are visible without overlapping the axis spine.
_BOUND_VIEW_MARGIN = 0.03


def _add_bound_lines_1d(ax: plt.Axes, lb: float, ub: float) -> None:
    """Draw dashed vertical lines at the lower and upper parameter bounds."""
    for bound in (lb, ub):
        ax.axvline(
            bound,
            color="0.5",
            linestyle="--",
            linewidth=1.4,
            alpha=0.95,
            zorder=1,
        )


def _add_bound_lines_2d(
    ax: plt.Axes,
    x_lb: float,
    x_ub: float,
    y_lb: float,
    y_ub: float,
) -> None:
    """Draw dashed lines at the lower and upper bounds on both axes."""
    for bound in (x_lb, x_ub):
        ax.axvline(
            bound,
            color="0.5",
            linestyle="--",
            linewidth=1.4,
            alpha=0.95,
            zorder=1,
        )
    for bound in (y_lb, y_ub):
        ax.axhline(
            bound,
            color="0.5",
            linestyle="--",
            linewidth=1.4,
            alpha=0.95,
            zorder=1,
        )


def _add_panel_legend(
    ax: plt.Axes,
    handles: list[Line2D],
    fontsize: int,
    loc: str = "upper left",
) -> None:
    """Add a compact styled legend to a subplot."""
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    label_to_handle = {
        label: handle
        for handle, label in zip(
            existing_handles, existing_labels, strict=True
        )
        if label
    }
    for handle in handles:
        if handle.get_label() not in label_to_handle:
            label_to_handle[handle.get_label()] = handle
    ax.legend(
        label_to_handle.values(),
        label_to_handle.keys(),
        loc=loc,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.85",
        fontsize=fontsize,
        handlelength=1.8,
        borderpad=0.4,
    )


def profiles(
    results: Result | Sequence[Result],
    ax=None,
    profile_indices: Sequence[int] = None,
    size: tuple[float, float] = (18.5, 6.5),
    reference: ReferencePoint | Sequence[ReferencePoint] = None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: Sequence[str] = None,
    x_labels: Sequence[str] = None,
    profile_list_ids: int | Sequence[int] = 0,
    ratio_min: float = 0.0,
    show_bounds: bool = False,
    plot_objective_values: bool = False,
    quality_colors: bool = False,
) -> plt.Axes:
    """
    Plot classical 1D profile plot.

    Using the posterior, e.g. Gaussian like profile.

    Parameters
    ----------
    results:
        List of or single `pypesto.Result` after profiling.
    ax:
        List of axes objects to use.
    profile_indices:
        List of integer values specifying which profiles should be plotted.
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified.
    reference:
        List of reference points for optimization results, containing at
        least a function value fval.
    colors:
        List of colors, or single color. If multiple colors are passed, their
        number needs to correspond to either the number of results or the
        number of profile_list_ids. Cannot be provided if quality_colors is set to True.
    legends:
        Labels for line plots, one label per result object.
    x_labels:
        Labels for parameter value axes (e.g. parameter names).
    profile_list_ids:
        Index or list of indices of the profile lists to visualize.
    ratio_min:
        Minimum ratio below which to cut off.
    show_bounds:
        Whether to show, and extend the plot to, the lower and upper bounds.
    plot_objective_values:
        Whether to plot the objective function values instead of the likelihood
        ratio values.
    quality_colors:
        If set to True, the profiles are colored according to types of steps the
        profiler took. This gives additional information about the profile quality.
        Red indicates a step for which min_step_size was reduced, blue indicates a step for which
        max_step_size was increased, and green indicates a step for which the profiler
        had to resample the parameter vector due to optimization failure of the previous two.
        Black indicates a step for which none of the above was necessary. This option is only
        available if there is only one result and one profile_list_id (one profile per plot).

    Returns
    -------
    ax:
        The plot axes.
    """

    if colors is not None and quality_colors:
        raise ValueError(
            "Cannot visualize the profiles with `quality_colors` of profiler_result.color_path "
            " and `colors` provided at the same time. Please provide only one of them."
        )

    # parse input
    results, profile_list_ids, colors, legends = process_result_list_profiles(
        results, profile_list_ids, legends, colors
    )

    # get the parameter ids to be plotted
    profile_indices = process_profile_indices(
        results, profile_indices, profile_list_ids
    )

    # loop over results
    for i_result, result in enumerate(results):
        for i_profile_list, profile_list_id in enumerate(profile_list_ids):
            fvals, color_paths = handle_inputs(
                result,
                profile_indices=profile_indices,
                profile_list=profile_list_id,
                ratio_min=ratio_min,
                plot_objective_values=plot_objective_values,
            )

            # add x_labels for parameters
            if x_labels is None:
                x_labels = [
                    name
                    for name, fval in zip(
                        result.problem.x_names, fvals, strict=True
                    )
                    if fval is not None
                ]

            # plot multiple results or profile runs into one figure?
            if len(results) == 1 and len(profile_list_ids) > 1:
                # multiple profile runs per axes object
                color_ind = i_profile_list
            else:
                # multiple results per axes object
                color_ind = i_result

            # If quality_colors is set to True, we use the colors provided
            # by profiler_result.color_path. This will be done only if there is
            # only one result and one profile_list_id (basically one profile per plot).
            if (
                len(results) == 1
                and len(profile_list_ids) == 1
                and quality_colors
            ):
                color = color_paths
            else:
                color = colors[color_ind]

            # call lowlevel routine
            ax = profiles_lowlevel(
                fvals=fvals,
                ax=ax,
                size=size,
                color=color,
                legend_text=legends[color_ind],
                x_labels=x_labels,
                show_bounds=show_bounds,
                lb_full=result.problem.lb_full,
                ub_full=result.problem.ub_full,
                plot_objective_values=plot_objective_values,
            )

    # parse and apply plotting options
    ref = create_references(references=reference)

    # plot reference points
    ax = handle_reference_points(ref, ax, profile_indices)

    plt.tight_layout()

    return ax


def profiles_lowlevel(
    fvals: float | Sequence[float],
    ax: Sequence[plt.Axes] | None = None,
    size: tuple[float, float] = (18.5, 6.5),
    color: COLOR | list[np.ndarray] | None = None,
    legend_text: str = None,
    x_labels=None,
    show_bounds: bool = False,
    lb_full: Sequence[float] = None,
    ub_full: Sequence[float] = None,
    plot_objective_values: bool = False,
) -> list[plt.Axes]:
    """
    Lowlevel routine for profile plotting.

    Working with a list of arrays only, opening different axes objects in case.

    Parameters
    ----------
    fvals:
        Values to plot.
    ax:
        List of axes object to use.
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified.
    color:
        Color for profiles in plot. In case of quality_colors=True, this is a list of
        np.ndarray[RGBA] for each profile -- one color per profile point for each profile.
    legend_text:
        Label for line plots.
    show_bounds:
        Whether to show, and extend the plot to, the lower and upper bounds.
    lb_full:
        Lower bound.
    ub_full:
        Upper bound.
    plot_objective_values:
        Whether to plot the objective function values instead of the likelihood
        ratio values.

    Returns
    -------
    The plot axes.
    """
    # axes
    if ax is None:
        ax = []
        fig = plt.figure()
        fig.set_size_inches(*size)
        create_new_ax = True
    else:
        plt.axes(ax[0])
        fig = plt.gcf()
        create_new_ax = False

    # count number of necessary axes
    if isinstance(fvals, Sequence):
        n_fvals = len(fvals)
    else:
        n_fvals = 1
        fvals = [fvals]

    # number of non-trivial profiles
    n_profiles = sum(fval is not None for fval in fvals)

    # if axes already exists, we have to match profiles to axes
    if not create_new_ax:
        if n_fvals != len(ax) and n_profiles != len(ax):
            raise ValueError(
                "Number of axes does not match number of profiles. Stopping."
            )
        elif n_fvals == len(ax) and n_profiles != len(ax):
            # we may have some empty profiles, which we have to skip
            n_plots = n_fvals
        else:
            # n_profiles == len(ax):, we have exactly as many profiles as axes
            n_plots = n_profiles
    else:
        n_plots = n_profiles

    if lb_full is None:
        lb_full = [None] * len(fvals)
    if ub_full is None:
        ub_full = [None] * len(fvals)

    # compute number of columns and rows
    columns = np.ceil(np.sqrt(n_plots))
    if n_plots > columns * (columns - 1):
        rows = columns
    else:
        rows = columns - 1

    counter = 0
    for i_plot, (fval, lb, ub) in enumerate(
        zip(fvals, lb_full, ub_full, strict=True)
    ):
        # if we have empty profiles and more axes than profiles: skip
        if n_plots != n_fvals and fval is None:
            continue
        # If we use colors from profiler_result.color_path,
        # we need to take the color path of each profile
        if isinstance(color, list) and isinstance(color[i_plot], np.ndarray):
            color_i = color[i_plot]
        else:
            color_i = color

        # handle legend
        if i_plot == 0:
            tmp_legend = legend_text
        else:
            tmp_legend = None

        # create or choose an axes object
        if create_new_ax:
            ax.append(fig.add_subplot(int(rows), int(columns), counter + 1))
        else:
            plt.axes(ax[counter])

        # plot if data
        if fval is not None:
            # run lowlevel routine for one profile
            ax[counter] = profile_lowlevel(
                fval,
                ax[counter],
                size=size,
                color=color_i,
                legend_text=tmp_legend,
                show_bounds=show_bounds,
                lb=lb,
                ub=ub,
            )

        # labels
        if x_labels is None:
            ax[counter].set_xlabel(f"Parameter {i_plot}")
        else:
            ax[counter].set_xlabel(x_labels[counter])

        if counter % columns == 0:
            if plot_objective_values:
                ax[counter].set_ylabel("Objective function value")
            else:
                ax[counter].set_ylabel("Log-posterior ratio")

        # increase counter and cleanup legend
        counter += 1

    return ax


def profile_lowlevel(
    fvals: Sequence[float],
    ax: plt.Axes | None = None,
    size: tuple[float, float] = (18.5, 6.5),
    color: COLOR | np.ndarray | None = None,
    legend_text: str = None,
    show_bounds: bool = False,
    lb: float = None,
    ub: float = None,
) -> plt.Axes:
    """
    Lowlevel routine for plotting one profile, working with a numpy array only.

    Parameters
    ----------
    fvals:
        Values to plot.
    ax:
        Axes object to use.
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified.
    color:
        Color for profiles in plot. A single color or an array of RGBA for each profile point
    legend_text:
        Label for line plots.
    show_bounds:
        Whether to show, and extend the plot to, the lower and upper bounds.
    lb:
        Lower bound.
    ub:
        Upper bound.

    Returns
    -------
    The plot axes.
    """
    # parse input
    fvals = np.asarray(fvals)
    # get colors
    if color is None or is_color_like(color):
        color = assign_colors([1.0], color)
        single_color = True
    else:
        single_color = False

    # axes
    if ax is None:
        ax = plt.subplots()[1]
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Log-posterior ratio")
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # plot
    if fvals.size != 0:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        xs = fvals[0, :]
        ratios = fvals[1, :]

        # If we use colors from profiler_result.color_path,
        # we need to make a mapping from profile points to their colors
        if not single_color:
            # Create a mapping from (x, ratio) to color
            point_to_color = dict(
                zip(zip(xs, ratios, strict=True), color, strict=True)
            )
        else:
            point_to_color = None

        # Plot each profile point individually to allow for different colors
        for i in range(1, len(xs)):
            point_color = (
                color[0]
                if single_color
                else tuple(point_to_color[(xs[i], ratios[i])])
            )
            ax.plot(
                [xs[i - 1], xs[i]],
                [ratios[i - 1], ratios[i]],
                color=color[0] if single_color else (0, 0, 0, 1),
                linestyle="-",
            )
            if not single_color and point_color != (0, 0, 0, 1):
                ax.plot(xs[i], ratios[i], color=point_color, marker="o")
            else:
                ax.plot(xs[i], ratios[i], color=point_color, marker=".")

        # Plot legend text
        ax.plot([], [], color=color[0], label=legend_text)

    if legend_text is not None:
        ax.legend()

    if show_bounds:
        ax.set_xlim([lb, ub])

    return ax


def handle_reference_points(ref, ax, profile_indices):
    """
    Handle reference points.

    Parameters
    ----------
    ref: list, optional
        List of reference points for optimization results, containing et
        least a function value fval
    ax: matplotlib.Axes, optional
        Axes object to use.
    profile_indices: list of integer values
        List of integer values specifying which profiles should be plotted.
    """
    if len(ref) > 0:
        # loop over axes objects
        for i_par, i_ax in enumerate(ax):
            for i_ref in ref:
                current_x = i_ref["x"][profile_indices[i_par]]
                i_ax.plot(
                    [current_x, current_x],
                    [0.0, 1.0],
                    color=i_ref.color,
                    label=i_ref.legend,
                )

            # create legend for reference points
            if i_ref.legend is not None:
                i_ax.legend()

    return ax


def handle_inputs(
    result: Result,
    profile_indices: Sequence[int],
    profile_list: int,
    ratio_min: float,
    plot_objective_values: bool,
) -> tuple[list, list]:
    """
    Retrieve the values of the profiles to be plotted.

    Parameters
    ----------
    result:
        Profile result obtained by 'profile.py'.
    profile_indices:
        Sequence of integer values specifying which profiles should be plotted.
    profile_list:
        Index of the profile list to be used for profiling.
    ratio_min:
        Exclude values where profile likelihood ratio is smaller than
        ratio_min.
    plot_objective_values:
        Whether to plot the objective function values instead of the likelihood

    Returns
    -------
    List of parameter values and ratios that need to be plotted.
    """
    # extract ratio values from result
    fvals = []
    colors = []
    for i_par in range(0, len(result.profile_result.list[profile_list])):
        if (
            i_par in profile_indices
            and result.profile_result.list[profile_list][i_par] is not None
        ):
            xs = result.profile_result.list[profile_list][i_par].x_path[
                i_par, :
            ]
            ratios = result.profile_result.list[profile_list][
                i_par
            ].ratio_path[:]
            colors_for_par = result.profile_result.list[profile_list][
                i_par
            ].color_path

            # constrain
            indices = np.where(ratios > ratio_min)
            xs = xs[indices]
            ratios = ratios[indices]
            colors_for_par = colors_for_par[indices]

            if plot_objective_values:
                obj_vals = result.profile_result.list[profile_list][
                    i_par
                ].fval_path
                obj_vals = obj_vals[indices]
                fvals_for_par = np.array([xs, obj_vals])
            else:
                fvals_for_par = np.array([xs, ratios])
        else:
            fvals_for_par = None
            colors_for_par = None
        fvals.append(fvals_for_par)
        colors.append(colors_for_par)

    return fvals, colors


def process_result_list_profiles(
    results: Result | list[Result],
    profile_list_ids: int | Sequence[int] | None,
    legends: str | list[str],
    colors: COLOR | list[COLOR] | np.ndarray | None = None,  # todo: check
) -> tuple[list[Result], list[int] | Sequence[int], list, list[str]]:
    """
    Assign colors and legends to a list of results.

    Takes also care of the special cases for profile plotting.

    Parameters
    ----------
    results:
        List of or single `pypesto.Result` after profiling.
    profile_list_ids:
        Index or list of indices of the profile lists to be used for profiling.
    colors:
        list of colors for plotting.
    legends:
        Legends for plotting

    Returns
    -------
    profile_indices: list of integer values
        corrected list of integer values specifying which profiles should be
        plotted.
    """
    # ensure list of ids
    if isinstance(profile_list_ids, int):
        profile_list_ids = [profile_list_ids]

    # check if we have a single result
    if isinstance(results, list):
        if len(results) != 1:
            # if we have no single result, then use the standard api
            results, colors, legends = process_result_list(
                results, colors, legends
            )
            return results, profile_list_ids, colors, legends
    else:
        # a single results was provided, so make a list out of it
        results = [results]

    # If we have a single result, we may still have multiple profile_list_ids
    # which should be plotted separately: use profile_list_ids as results dummy
    _, colors, legends = process_result_list(profile_list_ids, colors, legends)

    return results, profile_list_ids, colors, legends


def process_profile_indices(
    results: Sequence[Result],
    profile_indices: Sequence[int],
    profile_list_ids: int | Sequence[int],
):
    """
    Clean up profile_indices to be plotted.

    Retrieve the indices of the parameter for which profiles should be
    plotted later from a list of pypesto.ProfileResult objects.
    """
    # get all parameter indices, for which profiles were computed
    plottable_indices = set()
    for result in results:
        for profile_list_id in profile_list_ids:
            # get parameter indices, for which profiles were computed
            if profile_list_id < len(result.profile_result.list):
                tmp_indices = [
                    par_id
                    for par_id, prof in enumerate(
                        result.profile_result.list[profile_list_id]
                    )
                    if prof is not None
                ]
                # profile_indices should contain all parameter indices,
                # for which in at least one of the results a profile exists
                plottable_indices.update(tmp_indices)
    plottable_indices = sorted(plottable_indices)

    # get the profiles, which should be plotted and sanitize, if not plottable
    if profile_indices is None:
        profile_indices_ret = list(plottable_indices)
    else:
        profile_indices_ret = list(profile_indices)
        for ind in profile_indices:
            if ind not in plottable_indices:
                profile_indices_ret.remove(ind)
                warn(
                    f"Requested to plot profile for parameter index {ind}, "
                    "but profile has not been computed.",
                    stacklevel=2,
                )

    return profile_indices_ret


def profile_lowlevel_2d(
    result: Result,
    profile_index: int,
    second_par_index: int,
    ax: plt.Axes,
    profile_list_id: int = 0,
    ratio_min: float = 0.0,
    cmap: str = "viridis",
    plot_objective_values: bool = False,
    x_labels: Sequence[str] = None,
    vmin: float = None,
    vmax: float = None,
) -> plt.Axes:
    """
    Lowlevel routine for plotting a two-parameter profile visualization.

    Visualizes the profile of one parameter (x-axis) while showing the values
    of a second parameter (y-axis), with colors indicating the objective ratio
    or function value. Axis limits are always set to the parameter bounds,
    with dashed lines marking the lower and upper bounds.
    Axis labels include the parameter scale (e.g. ``log10(k1)``) unless
    overridden via ``x_labels``.

    Parameters
    ----------
    result:
        A single `pypesto.Result` after profiling.
    profile_index:
        Integer index specifying which profile to plot (x-axis parameter).
    second_par_index:
        Integer index specifying which parameter to show on y-axis.
    ax:
        Axes object to use for plotting.
    profile_list_id:
        Index of the profile list to visualize.
    ratio_min:
        Minimum ratio below which to cut off.
    cmap:
        Colormap to use for the objective ratio/value colors.
    plot_objective_values:
        Whether to plot the objective function values instead of the likelihood
        ratio values.
    x_labels:
        Labels for the parameters (indexed by full parameter index).
        If None, labels are auto-generated from parameter names and scales.
    vmin:
        Minimum value for the color scale. If None, auto-scaled to the data.
    vmax:
        Maximum value for the color scale. If None, auto-scaled to the data.

    Returns
    -------
    The plot axes.
    """
    if result.profile_result is None:
        raise ValueError("Result does not contain profile results.")

    profile_list = result.profile_result.list[profile_list_id]

    if profile_list[profile_index] is None:
        raise ValueError(
            f"Profile for parameter {profile_index} has not been computed."
        )

    profiler_result = profile_list[profile_index]

    x_path = profiler_result.x_path
    ratio_path = profiler_result.ratio_path
    fval_path = profiler_result.fval_path

    x_values = x_path[profile_index, :]
    y_values = x_path[second_par_index, :]
    color_values = fval_path if plot_objective_values else ratio_path

    # Filter based on ratio_min
    indices = np.where(ratio_path >= ratio_min)
    x_values = x_values[indices]
    y_values = y_values[indices]
    color_values = color_values[indices]

    # Draw the connector line in profile traversal order (pre-sort) so it
    # represents the actual profile path rather than a color-sorted spaghetti.
    ax.plot(x_values, y_values, "k-", alpha=0.2, linewidth=0.8, zorder=0)

    # Draw best points on top: ascending for ratio (high on top),
    # descending for objective value (low on top).
    sort_idx = (
        np.argsort(-color_values)
        if plot_objective_values
        else np.argsort(color_values)
    )
    ax.scatter(
        x_values[sort_idx],
        y_values[sort_idx],
        c=color_values[sort_idx],
        cmap=cmap,
        s=30,
        vmin=vmin,
        vmax=vmax,
    )

    def _label(idx):
        if x_labels is not None:
            return x_labels[idx]
        return _parameter_label(result.problem, idx)

    ax.set_xlabel(_label(profile_index))
    ax.set_ylabel(_label(second_par_index))

    x_lb = result.problem.lb_full[profile_index]
    x_ub = result.problem.ub_full[profile_index]
    y_lb = result.problem.lb_full[second_par_index]
    y_ub = result.problem.ub_full[second_par_index]
    x_margin = _BOUND_VIEW_MARGIN * (x_ub - x_lb)
    y_margin = _BOUND_VIEW_MARGIN * (y_ub - y_lb)
    ax.set_xlim([x_lb - x_margin, x_ub + x_margin])
    ax.set_ylim([y_lb - y_margin, y_ub + y_margin])
    _add_bound_lines_2d(ax, x_lb, x_ub, y_lb, y_ub)

    return ax


def visualize_2d_profile(
    result: Result,
    profile_indices: Sequence[int] = None,
    size: tuple[float, float] = None,
    profile_list_id: int = 0,
    ratio_min: float = 0.0,
    cmap: str = "viridis",
    plot_objective_values: bool = False,
    x_labels: Sequence[str] = None,
    profile_color: COLOR | np.ndarray | None = None,
    reference: ReferencePoint | Sequence[ReferencePoint] = None,
    label_fontsize: int = 14,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create an n×n grid of profile plots.

    Diagonal plots show 1D profiles (likelihood ratio vs. parameter value).
    Off-diagonal plots show the path of one parameter while another is
    profiled, with color indicating the likelihood ratio or objective value.
    Legend panels summarizing profile points and bound lines are drawn on
    the top-left diagonal and the first off-diagonal subplot.

    Parameters
    ----------
    result:
        A single `pypesto.Result` after profiling.
    profile_indices:
        List of integer indices specifying which parameters to include.
        If None, all parameters with computed profiles are included.
    size:
        Figure size (width, height) in inches. If None, automatically sized
        based on number of parameters (3.5 inches per parameter).
    profile_list_id:
        Index of the profile list to visualize.
    ratio_min:
        Minimum ratio below which to cut off.
    cmap:
        Colormap to use for the 2D off-diagonal scatter plots.
    plot_objective_values:
        Whether to plot the objective function values instead of the likelihood
        ratio values.
    x_labels:
        Labels for the parameters (indexed by full parameter index).
        If None, labels are auto-generated from parameter names and scales.
    profile_color:
        Color for the diagonal 1D profile lines. Passed directly to
        :func:`profile_lowlevel`. If None, the default color is used.
    reference:
        List of reference points for optimization results, shown on diagonal
        1D plots.
    label_fontsize:
        Font size for axis labels and the colorbar label. Tick labels are
        drawn two points smaller.

    Returns
    -------
    fig:
        The figure object.
    axes:
        Array of axes objects (n×n grid).
    """
    if result.profile_result is None:
        raise ValueError("Result does not contain profile results.")

    profile_list = result.profile_result.list[profile_list_id]

    if profile_indices is None:
        profile_indices = [
            i for i, prof in enumerate(profile_list) if prof is not None
        ]

    n_params = len(profile_indices)

    if n_params == 0:
        raise ValueError("No profiles available to plot.")

    if size is None:
        # +1 inch of extra width reserves space for the colorbar so that
        # each subplot cell remains approximately square.
        size = (n_params * 3.5 + 1, n_params * 3.5)

    fig, axes = plt.subplots(
        n_params,
        n_params,
        figsize=size,
        constrained_layout=True,
    )
    fig.get_layout_engine().set(wspace=0.1, hspace=0.1)

    if n_params == 1:
        axes = np.array([[axes]])

    ref = create_references(references=reference)

    def _label(idx):
        if x_labels is not None:
            return x_labels[idx]
        return _parameter_label(result.problem, idx)

    # Compute global color range across all 2D off-diagonal subplots so the
    # shared colorbar is accurate for every panel.
    all_color_values = []
    for row_idx in profile_indices:
        for col_idx in profile_indices:
            if row_idx == col_idx or profile_list[col_idx] is None:
                continue
            profiler = profile_list[col_idx]
            mask = profiler.ratio_path >= ratio_min
            vals = (
                profiler.fval_path[mask]
                if plot_objective_values
                else profiler.ratio_path[mask]
            )
            if vals.size > 0:
                all_color_values.append(vals)
    if all_color_values:
        all_vals = np.concatenate(all_color_values)
        color_vmin, color_vmax = float(all_vals.min()), float(all_vals.max())
    else:
        color_vmin, color_vmax = None, None

    first_2d_ax = None
    last_2d_ax = None

    for i, row_param_idx in enumerate(profile_indices):
        for j, col_param_idx in enumerate(profile_indices):
            ax = axes[i, j]

            if i == j:
                # Diagonal: 1D profile
                fvals, _ = handle_inputs(
                    result,
                    profile_indices=[row_param_idx],
                    profile_list=profile_list_id,
                    ratio_min=ratio_min,
                    plot_objective_values=plot_objective_values,
                )

                if fvals[row_param_idx] is not None:
                    profile_lowlevel(
                        fvals[row_param_idx],
                        ax,
                        show_bounds=True,
                        color=profile_color,
                        lb=result.problem.lb_full[row_param_idx],
                        ub=result.problem.ub_full[row_param_idx],
                    )
                    # Fix integer tick locator from profile_lowlevel for float params
                    ax.xaxis.set_major_locator(plt.AutoLocator())
                    ax.set_xlabel(_label(row_param_idx))
                    ax.set_ylabel(
                        "Objective value"
                        if plot_objective_values
                        else "Log-posterior ratio"
                    )
                    diag_lb = result.problem.lb_full[row_param_idx]
                    diag_ub = result.problem.ub_full[row_param_idx]
                    diag_margin = _BOUND_VIEW_MARGIN * (diag_ub - diag_lb)
                    ax.set_xlim([diag_lb - diag_margin, diag_ub + diag_margin])
                    _add_bound_lines_1d(ax, diag_lb, diag_ub)

                    if len(ref) > 0:
                        for i_ref in ref:
                            current_x = i_ref["x"][row_param_idx]
                            ax.plot(
                                [current_x, current_x],
                                [0.0, 1.0],
                                color=i_ref.color,
                                label=i_ref.legend
                                if i == 0 and j == 0
                                else None,
                            )
                    if i == 0 and j == 0:
                        profile_legend_color = (
                            profile_color
                            if profile_color is not None
                            else "red"
                        )
                        _add_panel_legend(
                            ax,
                            handles=[
                                Line2D(
                                    [0],
                                    [0],
                                    color=profile_legend_color,
                                    linewidth=2.0,
                                    label="Profile",
                                ),
                                Line2D(
                                    [0],
                                    [0],
                                    color="0.65",
                                    linestyle="--",
                                    linewidth=1.0,
                                    label="Bounds",
                                ),
                            ],
                            fontsize=label_fontsize - 3,
                        )

            else:
                # Off-diagonal: 2D profile
                # subplot (i, j): x-axis = col_param_idx, y-axis = row_param_idx
                try:
                    profile_lowlevel_2d(
                        result=result,
                        profile_index=col_param_idx,
                        second_par_index=row_param_idx,
                        ax=ax,
                        profile_list_id=profile_list_id,
                        ratio_min=ratio_min,
                        cmap=cmap,
                        plot_objective_values=plot_objective_values,
                        x_labels=x_labels,
                        vmin=color_vmin,
                        vmax=color_vmax,
                    )
                    if first_2d_ax is None:
                        first_2d_ax = ax
                    last_2d_ax = ax
                except (ValueError, IndexError):
                    ax.text(
                        0.5,
                        0.5,
                        "No profile",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

    # yaxis.labelpad is tightened so y-labels stay close to their axis
    # rather than floating in the gap between columns.
    for ax in axes.flat:
        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_weight("bold")
        ax.yaxis.labelpad = 2
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=label_fontsize - 2)

    if first_2d_ax is not None:
        _add_panel_legend(
            first_2d_ax,
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor="0.35",
                    markeredgecolor="none",
                    markersize=6,
                    label="Profile points",
                ),
                Line2D(
                    [0],
                    [0],
                    color="0.65",
                    linestyle="--",
                    linewidth=1.0,
                    label="Bounds",
                ),
            ],
            fontsize=label_fontsize - 3,
        )

    if last_2d_ax is not None:
        scatter = last_2d_ax.collections[-1]
        cbar = fig.colorbar(scatter, ax=axes)
        cbar.set_label(
            "Objective value"
            if plot_objective_values
            else "Log-posterior ratio",
            rotation=270,
            labelpad=20,
            fontsize=label_fontsize,
            fontweight="bold",
        )
        cbar.ax.tick_params(labelsize=label_fontsize - 2)

    return fig, axes
