from __future__ import annotations

from collections.abc import Sequence
from warnings import warn

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import is_color_like
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ..C import COLOR, LABEL_LIKELIHOOD_RATIO, LABEL_OBJECTIVE
from ..problem import Problem
from ..profile import chi2_quantile_to_ratio
from ..result import Result
from .clust_color import assign_colors
from .misc import (
    get_ax,
    get_axes_array,
    hide_unused_axes,
    make_grid_shape,
    process_result_list,
)
from .reference_points import ReferencePoint, create_references
from ._style import (
    BOUND_VIEW_MARGIN,
    COLORBAR_WIDTH,
    GRID_SIZE_PER_COL,
    GRID_SIZE_PER_ROW,
    add_colorbar,
    draw_bounds_1d,
    draw_bounds_2d,
    resolve_style,
)


def _parameter_label(problem: Problem, idx: int) -> str:
    """Return a scale-aware axis label for parameter ``idx``."""
    name = problem.x_names[idx]
    scale = problem.x_scales[idx] if problem.x_scales is not None else "lin"
    if scale == "log10":
        return f"log10({name})"
    if scale == "log":
        return f"log({name})"
    return name


def profiles(
    results: Result | Sequence[Result],
    ax: matplotlib.axes.Axes | None = None,
    profile_indices: Sequence[int] | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = None,
    reference: ReferencePoint | Sequence[ReferencePoint] | None = None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: Sequence[str] | None = None,
    x_labels: Sequence[str] | None = None,
    profile_list_ids: int | Sequence[int] = 0,
    ratio_min: float = 0.0,
    confidence_level: float | None = 0.95,
    show_bounds: bool = True,
    show_mle: bool = True,
    plot_objective_values: bool = False,
    quality_colors: bool = False,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
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
    title:
        Figure title.
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
        Minimum likelihood-ratio value below which to cut off profile points.
        Mutually exclusive with ``confidence_level``.
    confidence_level:
        Confidence level in (0, 1) (e.g. ``0.95``) for the profile CI
        overlay. Pass ``None`` to suppress the CI.
    show_bounds:
        Whether to show, and extend the plot to, the lower and upper bounds.
    show_mle:
        Whether to mark the MLE (best optimizer result) on each profile panel.
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
    style_kwargs:
        Style overrides; forwarded unchanged to :func:`profile_lowlevel`.
        See that function's documentation for the supported keys, and
        :data:`pypesto.visualize._style._DEFAULTS` for the full list.

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

    if confidence_level is not None and ratio_min != 0.0:
        raise ValueError(
            "Pass either `confidence_level` or `ratio_min`, not both."
        )

    style = resolve_style(style_kwargs)

    # parse input
    results, profile_list_ids, colors, legends = process_result_list_profiles(
        results, profile_list_ids, legends, colors, style=style
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

            # scale-aware x labels (e.g. "log10(k1)")
            if x_labels is None:
                x_labels = [
                    _parameter_label(result.problem, i_par)
                    for i_par, fval in enumerate(fvals)
                    if fval is not None
                ]

            # plot multiple results or profile runs into one figure?
            if len(results) == 1 and len(profile_list_ids) > 1:
                color_ind = i_profile_list
            else:
                color_ind = i_result

            if (
                len(results) == 1
                and len(profile_list_ids) == 1
                and quality_colors
            ):
                color = color_paths
            else:
                color = colors[color_ind]

            # MLE x per parameter — only for first result/profile_list
            x_mle_full = None
            if show_mle and i_result == 0 and i_profile_list == 0:
                if (
                    result.optimize_result is not None
                    and len(result.optimize_result.list) > 0
                    and result.optimize_result.list[0].x is not None
                ):
                    x_mle_full = result.optimize_result.list[0].x

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
                confidence_level=confidence_level,
                x_mle_full=x_mle_full,
                show_mle=show_mle,
                title=title,
                style_kwargs=style_kwargs,
            )

    # parse and apply plotting options
    ref = create_references(references=reference)

    # plot reference points
    ax = handle_reference_points(ref, ax, profile_indices)

    return ax


def profiles_lowlevel(
    fvals: float | Sequence[float],
    ax: Sequence[matplotlib.axes.Axes] | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = None,
    color: COLOR | list[np.ndarray] | None = None,
    legend_text: str | None = None,
    x_labels: Sequence[str] | None = None,
    show_bounds: bool = True,
    lb_full: Sequence[float] | None = None,
    ub_full: Sequence[float] | None = None,
    plot_objective_values: bool = False,
    confidence_level: float | None = None,
    x_mle_full: Sequence[float] | None = None,
    show_mle: bool = True,
    style_kwargs: dict | None = None,
) -> list[matplotlib.axes.Axes]:
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
    title:
        Figure title.
    color:
        Color for profiles in plot. In case of quality_colors=True, this is a list of
        np.ndarray[RGBA] for each profile -- one color per profile point for each profile.
    legend_text:
        Label for line plots.
    x_labels:
        Labels for the per-panel x-axes. If ``None``, default placeholder
        labels are used.
    show_bounds:
        Whether to show, and extend the plot to, the lower and upper bounds.
    lb_full:
        Lower bound.
    ub_full:
        Upper bound.
    plot_objective_values:
        Whether to plot the objective function values instead of the likelihood
        ratio values.
    confidence_level:
        Confidence level in (0, 1) for the CI overlay. Pass ``None`` to
        suppress.
    x_mle_full:
        MLE values per full-parameter index. When provided and ``show_mle``
        is ``True``, an MLE marker is drawn on each panel.
    show_mle:
        Whether to draw the MLE marker on each panel.
    style_kwargs:
        Style overrides; forwarded unchanged to :func:`profile_lowlevel`.
        See that function's documentation for the supported keys, and
        :data:`pypesto.visualize._style._DEFAULTS` for the full list.

    Returns
    -------
    The plot axes.
    """
    resolve_style(style_kwargs)

    # count number of necessary axes
    if isinstance(fvals, Sequence):
        n_fvals = len(fvals)
    else:
        n_fvals = 1
        fvals = [fvals]

    # number of non-trivial profiles
    n_profiles = sum(fval is not None for fval in fvals)

    # if axes already exists, we have to match profiles to axes
    if ax is not None:
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
    if x_mle_full is None:
        x_mle_full = [None] * len(fvals)

    # grid layout
    num_row, num_col = make_grid_shape(max(n_plots, 1))

    # axes — create a fresh grid unless the caller supplied one
    if ax is None:
        axes = get_axes_array(nrows=num_row, ncols=num_col, size=size)
        axes = hide_unused_axes(axes=axes, n_used=n_plots, clear=True)
        ax = list(axes.flat[:n_plots])

    counter = 0
    for i_plot, (fval, lb, ub, x_mle_i) in enumerate(
        zip(fvals, lb_full, ub_full, x_mle_full, strict=True)
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

        # plot if data
        if fval is not None:
            ax[counter] = profile_lowlevel(
                fval,
                ax[counter],
                color=color_i,
                legend_text=tmp_legend,
                show_bounds=show_bounds,
                lb=lb,
                ub=ub,
                confidence_level=confidence_level,
                x_mle=x_mle_i,
                show_mle=show_mle,
                show_legend=(counter == 0),
                title=None,
                style_kwargs=style_kwargs,
            )

        # x-label always; y-label only on leftmost column
        if x_labels is None:
            ax[counter].set_xlabel(f"Parameter {i_plot}")
        else:
            ax[counter].set_xlabel(x_labels[counter])

        if counter % num_col == 0:
            ax[counter].set_ylabel(
                LABEL_OBJECTIVE if plot_objective_values else LABEL_LIKELIHOOD_RATIO
            )
        else:
            ax[counter].set_ylabel("")

        counter += 1

    if title is not None and ax:
        ax[0].figure.suptitle(title)

    return ax


def profile_lowlevel(
    fvals: Sequence[float],
    ax: matplotlib.axes.Axes | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = "Profile likelihood",
    color: COLOR | np.ndarray | None = None,
    legend_text: str | None = None,
    x_label: str = "Parameter value",
    show_bounds: bool = True,
    lb: float | None = None,
    ub: float | None = None,
    confidence_level: float | None = None,
    x_mle: float | None = None,
    show_mle: bool = True,
    show_legend: bool = True,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
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
    title:
        Axes title. Pass ``None`` to suppress.
    color:
        Single color or per-point RGBA array (for ``quality_colors`` mode).
    legend_text:
        Label for the profile line in the legend.
    x_label:
        X-axis label. Defaults to ``"Parameter value"``.
    show_bounds:
        Whether to draw vertical bound lines at *lb* and *ub*.
    lb:
        Lower parameter bound.
    ub:
        Upper parameter bound.
    confidence_level:
        When given, draws a dashed threshold line at the corresponding
        likelihood ratio and a thick segment showing the CI range.
    x_mle:
        x-position of the MLE. When given and *show_mle* is ``True``, a dot
        is drawn at ``(x_mle, 1.0)``.
    show_mle:
        Whether to draw the MLE marker.
    show_legend:
        Whether to render the per-panel legend. The grid caller passes
        ``True`` only on the first panel so the legend appears once.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``line_color``, ``trace_linewidth``, ``trace_marker_size`` —
          profile line / point styling.
        - ``cmap_ci`` — colormap from which the CI threshold and range
          colours are sampled.
        - ``ci_linewidth`` — width of the CI range segment.
        - ``mle_color``, ``line_marker_size`` — MLE marker styling.
        - ``bound_color``, ``bound_linestyle``, ``bound_linewidth``,
          ``bound_alpha`` — parameter-bound line style.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    The plot axes.
    """
    style = resolve_style(style_kwargs)

    fvals = np.asarray(fvals)
    if color is None:
        color = assign_colors([1.0], style["line_color"], style=style)
        single_color = True
    elif is_color_like(color):
        color = assign_colors([1.0], color, style=style)
        single_color = True
    else:
        single_color = False

    ax = get_ax(ax, size)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(LABEL_LIKELIHOOD_RATIO)

    if fvals.size != 0:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        xs = fvals[0, :]
        ratios = fvals[1, :]

        if single_color:
            ax.plot(xs, ratios, color=color[0], linestyle="-",
                    linewidth=style["trace_linewidth"])
            ax.plot(xs, ratios, ".", color=color[0],
                    markersize=style["trace_marker_size"])
        else:
            # quality_colors: one color per profile point
            point_to_color = dict(
                zip(zip(xs, ratios, strict=True), color, strict=True)
            )
            for i in range(1, len(xs)):
                point_color = tuple(point_to_color[(xs[i], ratios[i])])
                ax.plot([xs[i - 1], xs[i]], [ratios[i - 1], ratios[i]],
                        color=(0, 0, 0, 1), linestyle="-")
                ax.plot(xs[i], ratios[i], color=point_color,
                        marker="o" if point_color != (0, 0, 0, 1) else ".")

        ax.plot([], [], color=color[0], label=legend_text or "Profile")

        if confidence_level is not None:
            ci_ratio = chi2_quantile_to_ratio(confidence_level)
            ci_color = mpl.colormaps[style["cmap_ci"]](0.7)
            # dashed threshold line
            ax.axhline(ci_ratio, color=ci_color, linestyle="--",
                       linewidth=0.9, alpha=0.8)
            # thick CI range segment(s)
            above = ratios >= ci_ratio
            if above.any():
                trans = np.diff(above.astype(int))
                starts = list(np.where(trans == 1)[0] + 1)
                ends = list(np.where(trans == -1)[0] + 1)
                if above[0]:
                    starts = [0] + starts
                if above[-1]:
                    ends = ends + [len(xs)]
                x_lo_lim, x_hi_lim = ax.get_xlim()
                for s, e in zip(starts, ends, strict=True):
                    # if clipped at start/end: profile extends beyond data →
                    # use axis limit so the segment reaches the edge
                    if s == 0:
                        t_lo = x_lo_lim
                    else:
                        t_lo = (
                            xs[s - 1] + (ci_ratio - ratios[s - 1])
                            / (ratios[s] - ratios[s - 1])
                            * (xs[s] - xs[s - 1])
                        )
                    if e == len(xs):
                        t_hi = x_hi_lim
                    else:
                        t_hi = (
                            xs[e - 1] + (ci_ratio - ratios[e - 1])
                            / (ratios[e] - ratios[e - 1])
                            * (xs[e] - xs[e - 1])
                        )
                    ax.hlines(ci_ratio, t_lo, t_hi, color=ci_color,
                              linewidth=style["ci_linewidth"])
            # legend proxies: dashed threshold line + thick CI range segment
            ax.plot([], [], color=ci_color, linestyle="--",
                    linewidth=0.9, alpha=0.8,
                    label=f"{confidence_level * 100:g}% threshold")
            ax.plot([], [], color=ci_color, linewidth=style["ci_linewidth"],
                    label=f"{confidence_level * 100:g}% CI")

        if show_mle and x_mle is not None and np.isfinite(x_mle):
            ax.plot(x_mle, 1.0, "o", color=style["mle_color"],
                    markersize=style["line_marker_size"], zorder=5,
                    label="MLE")

        ax.set_ylim(0.0, 1.1)

    if show_bounds and lb is not None and ub is not None:
        draw_bounds_1d(ax, lb, ub, axis="x", style=style)

    if show_legend:
        handles, labels = _ordered_profile_legend_items(ax)
        if handles:
            ax.legend(
                handles,
                labels,
                fontsize=9,
                handlelength=1.8,
                borderpad=0.4,
            )

    return ax


def _ordered_profile_legend_items(
    ax: matplotlib.axes.Axes,
) -> tuple[list, list[str]]:
    """Return deduplicated profile legend items in semantic order.

    Profile/run labels are placed last so multiple results appear together
    instead of being interleaved with the MLE and CI proxy artists.
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    if not by_label:
        return [], []

    def _rank(label: str) -> int:
        if label.endswith("threshold"):
            return 0
        if label.endswith("CI"):
            return 1
        if label == "MLE":
            return 2
        return 3

    ordered_labels = sorted(by_label, key=_rank)
    return [by_label[label] for label in ordered_labels], ordered_labels


def handle_reference_points(ref, ax, profile_indices):
    """
    Handle reference points.

    Parameters
    ----------
    ref: list, optional
        List of reference points for optimization results, containing et
        least a function value fval
    ax: matplotlib.axes.Axes, optional
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
    legends: str | list[str] | None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    style: dict | None = None,
) -> tuple[list[Result], Sequence[int], np.ndarray, list[str | None]]:
    """
    Assign colors and legends to a list of results.

    Takes also care of the special cases for profile plotting: a single
    result with multiple ``profile_list_ids`` is coloured per profile-list,
    while multiple results are coloured per result.

    Parameters
    ----------
    results:
        List of or single ``pypesto.Result`` after profiling.
    profile_list_ids:
        Index or list of indices of the profile lists to be used for
        profiling.
    legends:
        Legend label(s).  One per result, or one per profile-list when a
        single result with multiple profile lists is plotted.
    colors:
        RGBA colour list or single colour for plotting.  When ``None``,
        colours are assigned automatically.
    style:
        Resolved pyPESTO visualization style. Forwarded to
        :func:`process_result_list`.

    Returns
    -------
    results, profile_list_ids, colors, legends.
    """
    # ensure list of ids
    if isinstance(profile_list_ids, int):
        profile_list_ids = [profile_list_ids]

    # check if we have a single result
    if isinstance(results, list):
        if len(results) != 1:
            # if we have no single result, then use the standard api
            results, colors, legends = process_result_list(
                results, colors, legends, style=style
            )
            return results, profile_list_ids, colors, legends
    else:
        # a single results was provided, so make a list out of it
        results = [results]

    # If we have a single result, we may still have multiple profile_list_ids
    # which should be plotted separately: use profile_list_ids as results dummy
    _, colors, legends = process_result_list(
        profile_list_ids, colors, legends, style=style
    )

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
    ax: matplotlib.axes.Axes,
    title: str | None = "2D profile",
    profile_list_id: int = 0,
    ratio_min: float = 0.0,
    cmap: str = "viridis",
    plot_objective_values: bool = False,
    x_labels: Sequence[str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_bounds: bool = True,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
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
    title:
        Axes title. Pass ``None`` to suppress.
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
    show_bounds:
        Whether to draw the parameter-bound dashed lines and extend the
        axis limits to include the bounds. Default ``True``.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``line_color``, ``trace_linewidth``, ``trace_alpha`` — profile-path
          line styling.
        - ``scatter_size``, ``scatter_alpha``, ``scatter_linewidths``,
          ``scatter_edgecolors``, ``scatter_zorder`` — profile point geometry.
        - ``bound_color``, ``bound_linestyle``, ``bound_linewidth``,
          ``bound_alpha`` — parameter-bound line style.

    Returns
    -------
    The plot axes.
    """
    style = resolve_style(style_kwargs)
    if title is not None:
        ax.set_title(title)

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
    ax.plot(
        x_values,
        y_values,
        color=style["line_color"],
        alpha=style["trace_alpha"],
        linewidth=style["trace_linewidth"],
        zorder=0,
    )

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
        s=style["scatter_size"],
        alpha=style["scatter_alpha"],
        linewidths=style["scatter_linewidths"],
        edgecolors=style["scatter_edgecolors"],
        zorder=style["scatter_zorder"],
        vmin=vmin,
        vmax=vmax,
    )

    def _label(idx):
        if x_labels is not None:
            return x_labels[idx]
        return _parameter_label(result.problem, idx)

    ax.set_xlabel(_label(profile_index))
    ax.set_ylabel(_label(second_par_index))

    if show_bounds:
        x_lb = result.problem.lb_full[profile_index]
        x_ub = result.problem.ub_full[profile_index]
        y_lb = result.problem.lb_full[second_par_index]
        y_ub = result.problem.ub_full[second_par_index]
        draw_bounds_2d(ax, x_lb, x_ub, y_lb, y_ub, view_margin=True, style=style)

    return ax


def visualize_2d_profile(
    result: Result,
    profile_indices: Sequence[int] | None = None,
    axes: np.ndarray | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = None,
    profile_list_id: int = 0,
    ratio_min: float = 0.0,
    show_bounds: bool = True,
    plot_objective_values: bool = False,
    x_labels: Sequence[str] | None = None,
    profile_color: COLOR | np.ndarray | None = None,
    reference: ReferencePoint | Sequence[ReferencePoint] | None = None,
    style_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Create an n×n grid of profile plots.

    Diagonal plots show 1D profiles (likelihood ratio vs. parameter value).
    Off-diagonal plots show the path of one parameter while another is
    profiled, with color indicating the likelihood ratio or objective value.
    A single legend on the (0, 0) panel summarises Profile / Profile points
    / Bounds; the colorbar communicates the off-diagonal coloring.

    Layout and styling match :func:`optimization_scatter` and
    :func:`sampling_scatter`: shared per-column / per-row axis limits,
    optional ``show_bounds`` framing, and a colorbar sized via
    :func:`~pypesto.visualize._style.add_colorbar`.

    Parameters
    ----------
    result:
        A single `pypesto.Result` after profiling.
    profile_indices:
        List of integer indices specifying which parameters to include.
        If None, all parameters with computed profiles are included.
    axes:
        Optional axes grid to draw into. Must have shape
        ``(n_params, n_params)``.
    size:
        Figure size (width, height) in inches. If None, automatically sized
        from the shared grid dimensions plus colorbar width.
    title:
        Figure title.
    profile_list_id:
        Index of the profile list to visualize.
    ratio_min:
        Minimum ratio below which to cut off.
    show_bounds:
        Whether to draw the parameter-bound dashed lines and extend the
        axis limits to include the bounds. Default ``True``.
    plot_objective_values:
        Whether to plot the objective function values instead of the likelihood
        ratio values.
    x_labels:
        Labels for the parameters (indexed by full parameter index).
        If None, labels are auto-generated from parameter names and scales.
    profile_color:
        Color for the diagonal 1D profile lines. Passed directly to
        :func:`profile_lowlevel`. If None, the default ``line_color`` is used.
    reference:
        List of reference points for optimization results, shown on diagonal
        1D plots.
    style_kwargs:
        Style overrides. Keys used directly by this function:

        - ``cmap_posterior`` — colormap for the off-diagonal scatter (the
          profile ratio behaves like an offset posterior, so the posterior
          colormap is the natural fit).
        - ``line_color`` — default colour for the diagonal 1D profile
          line when ``profile_color`` is ``None``.
        - ``bound_color``, ``bound_linestyle``, ``bound_linewidth``,
          ``bound_alpha`` — parameter-bound line style.

        Additional keys are forwarded to :func:`profile_lowlevel` and
        :func:`profile_lowlevel_2d` and apply to the per-cell rendering.
        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    axes:
        2-D NumPy array of shape ``(n_params, n_params)`` containing one
        matplotlib Axes per panel.
    """
    style = resolve_style(style_kwargs)

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

    if size is None and axes is None:
        # grid panels + extra width for the shared colorbar
        size = (
            GRID_SIZE_PER_COL * n_params + COLORBAR_WIDTH,
            GRID_SIZE_PER_ROW * n_params,
        )

    axes = get_axes_array(axes=axes, nrows=n_params, ncols=n_params, size=size)
    fig = axes.flat[0].figure
    for ax in axes.flat:
        ax.clear()
        ax.set_visible(True)

    ref = create_references(references=reference)

    def _label(idx):
        if x_labels is not None:
            return x_labels[idx]
        return _parameter_label(result.problem, idx)

    cmap = style["cmap_posterior"]

    # Global color range across all 2D off-diagonal subplots so the shared
    # colorbar is accurate for every panel.
    all_color_values = []
    for col_idx in profile_indices:
        if profile_list[col_idx] is None:
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

    has_2d_panel = False

    # ------------------------------------------------------------------
    # Plot cells. Bounds and per-column / per-row limits are applied in
    # a second pass below so they're consistent across the grid.
    # ------------------------------------------------------------------
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
                        title=None,
                        show_bounds=False,
                        show_legend=False,
                        color=profile_color,
                        lb=result.problem.lb_full[row_param_idx],
                        ub=result.problem.ub_full[row_param_idx],
                        style_kwargs=style,
                    )
                    # Override profile_lowlevel's integer tick locator
                    ax.xaxis.set_major_locator(plt.AutoLocator())
                    ax.set_xlabel(_label(row_param_idx))
                    ax.set_ylabel(
                        LABEL_OBJECTIVE
                        if plot_objective_values
                        else LABEL_LIKELIHOOD_RATIO
                    )

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

            else:
                # Off-diagonal: 2D profile. subplot (i, j) → x = col_param_idx,
                # y = row_param_idx.
                try:
                    profile_lowlevel_2d(
                        result=result,
                        profile_index=col_param_idx,
                        second_par_index=row_param_idx,
                        ax=ax,
                        title=None,
                        profile_list_id=profile_list_id,
                        ratio_min=ratio_min,
                        cmap=cmap,
                        plot_objective_values=plot_objective_values,
                        x_labels=x_labels,
                        vmin=color_vmin,
                        vmax=color_vmax,
                        show_bounds=False,
                        style_kwargs=style,
                    )
                    has_2d_panel = True
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

    # ------------------------------------------------------------------
    # Per-column x-lims (shared down each column) and per-row y-lims
    # (shared across off-diagonal cells in each row). Matches the
    # optimization_scatter / sampling_scatter framing pattern.
    # ------------------------------------------------------------------
    col_xlims: dict[int, tuple[float, float]] = {}
    for j_pos, j in enumerate(profile_indices):
        if profile_list[j] is None:
            continue
        p = profile_list[j]
        mask = p.ratio_path >= ratio_min
        vals = p.x_path[j, mask]
        if vals.size == 0:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        if show_bounds:
            lo = min(lo, float(result.problem.lb_full[j]))
            hi = max(hi, float(result.problem.ub_full[j]))
        span = hi - lo
        pad = span * BOUND_VIEW_MARGIN if span > 0 else 0.5
        col_xlims[j_pos] = (lo - pad, hi + pad)

    row_ylims: dict[int, tuple[float, float]] = {}
    for i_pos, i in enumerate(profile_indices):
        vals_list = []
        for j_pos, j in enumerate(profile_indices):
            if i == j or profile_list[j] is None:
                continue
            p = profile_list[j]
            mask = p.ratio_path >= ratio_min
            vals_list.append(p.x_path[i, mask])
        if not vals_list:
            continue
        vals = np.concatenate(vals_list)
        if vals.size == 0:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        if show_bounds:
            lo = min(lo, float(result.problem.lb_full[i]))
            hi = max(hi, float(result.problem.ub_full[i]))
        span = hi - lo
        pad = span * BOUND_VIEW_MARGIN if span > 0 else 0.5
        row_ylims[i_pos] = (lo - pad, hi + pad)

    for j_pos, xlim in col_xlims.items():
        for i_pos in range(n_params):
            axes[i_pos, j_pos].set_xlim(xlim)
    for i_pos, ylim in row_ylims.items():
        for j_pos in range(n_params):
            if i_pos != j_pos:
                axes[i_pos, j_pos].set_ylim(ylim)

    # ------------------------------------------------------------------
    # Bound lines (after lim-setting so view_margin doesn't perturb).
    # ------------------------------------------------------------------
    if show_bounds:
        for i_pos, i in enumerate(profile_indices):
            for j_pos, j in enumerate(profile_indices):
                ax = axes[i_pos, j_pos]
                if i_pos == j_pos:
                    draw_bounds_1d(
                        ax,
                        result.problem.lb_full[i],
                        result.problem.ub_full[i],
                        axis="x",
                        view_margin=False,
                        style=style,
                    )
                else:
                    draw_bounds_2d(
                        ax,
                        result.problem.lb_full[j],
                        result.problem.ub_full[j],
                        result.problem.lb_full[i],
                        result.problem.ub_full[i],
                        view_margin=False,
                        style=style,
                    )

    # ------------------------------------------------------------------
    # Single legend on the (0, 0) panel.
    # ------------------------------------------------------------------
    profile_legend_color = (
        profile_color if profile_color is not None else style["line_color"]
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=profile_legend_color,
            linewidth=2.0,
            label="Profile",
        ),
    ]
    if has_2d_panel:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=plt.get_cmap(cmap)(0.5),
                markeredgecolor="none",
                markersize=6,
                label="Profile points",
            )
        )
    if show_bounds:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["bound_color"],
                linestyle=style["bound_linestyle"],
                linewidth=style["bound_linewidth"],
                alpha=style["bound_alpha"],
                label="Bounds",
            )
        )
    axes[0, 0].legend(handles=legend_handles)

    # ------------------------------------------------------------------
    # Shared colorbar.
    # ------------------------------------------------------------------
    if all_color_values:
        add_colorbar(
            fig,
            axes,
            np.concatenate(all_color_values),
            label=LABEL_OBJECTIVE
            if plot_objective_values
            else LABEL_LIKELIHOOD_RATIO,
            cmap=cmap,
            norm=mpl.colors.Normalize(vmin=color_vmin, vmax=color_vmax),
        )

    if title is not None:
        fig.suptitle(title)

    return axes
