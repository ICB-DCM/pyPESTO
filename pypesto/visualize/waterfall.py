from __future__ import annotations

from collections.abc import Sequence

import matplotlib.axes
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import inset_locator

from pypesto.util import delete_nan_inf

from ..C import ALL, COLOR, LABEL_OBJECTIVE, WATERFALL_MAX_VALUE
from ..result import Result
from .clust_color import assign_colors
from .misc import (
    get_ax,
    process_offset_y,
    process_result_list,
    process_start_indices,
    process_y_limits,
)
from .reference_points import ReferencePoint, create_references
from ._style import resolve_style


def waterfall(
    results: Result | Sequence[Result],
    ax: matplotlib.axes.Axes | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = "Waterfall plot",
    y_limits: tuple[float] | None = None,
    scale_y: str | None = "log10",
    offset_y: float | None = None,
    start_indices: Sequence[int] | int | None = None,
    n_starts_to_zoom: int = 0,
    reference: Sequence[ReferencePoint] | None = None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: Sequence[str] | str | None = None,
    order_by_id: bool = False,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot waterfall plot.

    Parameters
    ----------
    results:
        Optimization result obtained by 'optimize.py' or list of those
    ax: matplotlib.Axes, optional
        Axes object to use.
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified
    title:
        Axes title. Pass ``None`` to suppress.
    y_limits: float or ndarray, optional
        Maximum value to be plotted on the y-axis, or y-limits
    scale_y:
        May be logarithmic or linear ('log10' or 'lin')
    offset_y:
        Offset for the y-axis, if it is supposed to be in log10-scale
    start_indices:
        Integers specifying the multistart to be plotted or int specifying
        up to which start index should be plotted
    n_starts_to_zoom:
        Number of best multistarts that should be zoomed in.
        Should be smaller that the total number of multistarts
    reference:
        Reference points for optimization results, containing at least a
        function value fval
    colors:
        List of colors or single color for plotting. If not set, clustering is done
        and colors are assigned automatically
    legends:
        Labels for line plots, one label per result object
    order_by_id:
        Function values corresponding to the same start ID will be located at
        the same x-axis position. Only applicable when a list of result
        objects are provided. Default behavior is to sort the function values
        of each result independently of other results.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``cmap_discrete``, ``mle_color``, ``outlier_color`` — colours of
          the per-start scatter dots when clustering is used (best
          cluster, secondary clusters, isolated starts respectively).
        - ``ref_line_color`` — colour of the connecting waterfall line.
        - ``marker_size``, ``marker_linewidth`` — start-dot geometry.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    ax: matplotlib.Axes
        The plot axes.
    """
    ax = get_ax(ax, size)
    style = resolve_style(style_kwargs)

    if n_starts_to_zoom:
        # create zoom in
        inset_axes = inset_locator.inset_axes(
            ax, width="30%", height="30%", loc="center right"
        )
        inset_locator.mark_inset(ax, inset_axes, loc1=2, loc2=4)
    else:
        inset_axes = None

    # parse input
    (results, colors, legends) = process_result_list(
        results, colors, legends, style=style
    )

    # handle `order_by_id`
    if order_by_id:
        start_id_ordering = get_ordering_by_start_id(results)
        # Set start indices to all, and save actual start indices for later,
        # so that all fvals are retrieved by `process_offset_for_list`.
        # This enables use of `order_by_id` with `start_indices`.
        ordered_start_indices = process_start_indices(
            result=results[0], start_indices=start_indices
        )
        start_indices = ALL

    refs = create_references(references=reference)

    # precompute y-offset, if needed and if a list of results was passed
    fvals_all, offset_y = process_offset_for_list(
        offset_y, results, scale_y, start_indices, refs
    )

    # plotting routine needs the maximum number of multistarts
    max_len_fvals = 0

    # loop over results
    for j, fvals_raw in enumerate(fvals_all):
        # extract specific cost function values from result
        max_len_fvals = np.max([max_len_fvals, *fvals_raw.shape])

        # remove colors where value is infinite if colors were passed on
        if (
            colors[j] is not None
            and isinstance(colors[j], np.ndarray)
            and fvals_raw.size == colors[j].shape[0]
        ):
            colors[j] = colors[j][
                np.isfinite(np.transpose(fvals_raw)).flatten()
            ]

        # parse input
        if order_by_id:
            start_ids = [s.id for s in results[j].optimize_result.list]
            fvals_raw_is_finite = np.isfinite(fvals_raw) & (
                np.absolute(fvals_raw) < WATERFALL_MAX_VALUE
            )

            fvals = []
            for start_id in start_id_ordering:
                start_index = start_ids.index(start_id)
                if fvals_raw_is_finite[start_index]:
                    fvals.append(fvals_raw[start_index])
                else:
                    fvals.append(None)
            fvals = np.array(fvals)[ordered_start_indices]
        else:
            # remove nan or inf values in fvals
            # also remove extremely large values. These values result in `inf`
            # values in the output of `scipy.cluster.hierarchy.linkage` in the
            # method `pypesto.util.assign_clusters`, which raises errors.
            _, fvals = delete_nan_inf(
                fvals=fvals_raw, magnitude_bound=WATERFALL_MAX_VALUE
            )
            fvals.sort()

        # assign colors
        coloring = assign_colors(fvals, colors=colors[j], style=style)

        # call lowlevel plot routine
        ax = waterfall_lowlevel(
            fvals=fvals,
            scale_y=scale_y,
            offset_y=offset_y,
            ax=ax,
            size=size,
            colors=coloring,
            legend_text=legends[j],
            title=None,
            style_kwargs=style_kwargs,
        )

        if inset_axes is not None:
            inset_axes = waterfall_lowlevel(
                fvals=fvals[:n_starts_to_zoom],
                scale_y=scale_y,
                ax=inset_axes,
                colors=coloring[:n_starts_to_zoom],
                title=None,
                style_kwargs=style_kwargs,
            )
            # remove the title and axes labels for the zoom in subplot
            inset_axes.set(title=None, xlabel=None, ylabel=None)

    # apply changes specified be the user to the axis object
    ax = handle_options(ax, max_len_fvals, refs, y_limits, offset_y)
    if inset_axes is not None:
        inset_axes = handle_options(
            inset_axes, n_starts_to_zoom, refs, None, offset_y
        )

    if any(legends):
        ax.legend()
    # labels
    ax.set_xlabel("Ordered optimizer run")
    if offset_y == 0.0:
        ax.set_ylabel(LABEL_OBJECTIVE)
    else:
        ax.set_ylabel(f"{LABEL_OBJECTIVE} (offset={offset_y:0.3e})")
    if title is not None:
        ax.set_title(title)
    return ax


def waterfall_lowlevel(
    fvals,
    ax: matplotlib.axes.Axes | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = "Waterfall plot",
    scale_y: str = "log10",
    offset_y: float = 0.0,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legend_text: str | None = None,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot waterfall plot using list of function values.

    Parameters
    ----------
    fvals:
        Including values need to be plotted. `None` values indicate that the
        corresponding start index should be skipped.
    ax:
        Axes object to use.
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified
    title:
        Axes title. Pass ``None`` to suppress.
    scale_y:
        May be logarithmic or linear ('log10' or 'lin')
    offset_y:
        offset for the y-axis, if it is supposed to be in log10-scale
    colors:
        Color recognized by matplotlib or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically
    legend_text:
        Label for line plots
    style_kwargs:
        Style overrides; see :func:`waterfall` for the keys this plotter
        consumes. All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    ax: matplotlib.Axes
        The plot axes.
    """
    style = resolve_style(style_kwargs)
    ax = get_ax(ax, size)

    start_indices = [i for i, fval in enumerate(fvals) if fval is not None]
    fvals = [fvals[i] for i in start_indices]
    if colors is not None and colors.ndim == 2 and colors.shape[0] > 1:
        colors = [colors[i] for i in start_indices]

    # assign colors
    colors = assign_colors(fvals, colors=colors, style=style)

    # plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plot line
    if scale_y == "log10":
        ax.semilogy(start_indices, fvals, color=style["ref_line_color"])
        ax.set_yscale("log")
    else:
        ax.plot(start_indices, fvals, color=style["ref_line_color"])

    # Overlay with scatter points with individual colors
    # plotting in reverse order to ensure that the best points are plotted on top
    ax.scatter(
        start_indices[::-1],
        fvals[::-1],
        c=colors[::-1],
        marker="o",
        s=style["marker_size"],
        linewidth=style["marker_linewidth"],
        label=legend_text,
        zorder=2.0,
        alpha=1.0,
    )

    # check if y-axis has a reasonable scale
    y_min, y_max = ax.get_ylim()
    if scale_y == "log10":
        if np.log10(y_max) - np.log10(y_min) < 1.0:
            ax.set_ylim(
                ax.dataLim.y0 - 0.05 * abs(ax.dataLim.y0),
                ax.dataLim.y1 + 0.05 * abs(ax.dataLim.y1),
            )
    else:
        if y_max - y_min < 1.0:
            y_mean = 0.5 * (y_min + y_max)
            ax.set_ylim(y_mean - 0.5, y_mean + 0.5)

    # labels
    ax.set_xlabel("Ordered optimizer run")
    if offset_y == 0.0:
        ax.set_ylabel(LABEL_OBJECTIVE)
    else:
        ax.set_ylabel(f"{LABEL_OBJECTIVE} (offset={offset_y:0.3e})")
    if title is not None:
        ax.set_title(title)
    if legend_text is not None:
        ax.legend()

    return ax


def process_offset_for_list(
    offset_y: float,
    results: Sequence[Result],
    scale_y: str | None,
    start_indices: Sequence[int] | None = None,
    references: Sequence[ReferencePoint] | None = None,
) -> tuple[list[np.ndarray], float]:
    """
    Compute common offset_y and add it to `fvals` of results.

    Parameters
    ----------
    offset_y:
        User provided offset_y
    results:
        Optimization results obtained by 'optimize.py'
    scale_y:
        May be logarithmic or linear ('log10' or 'lin')
    start_indices:
        Integers specifying the multistart to be plotted or int specifying
        up to which start index should be plotted
    references:
        Reference points that will be plotted along with the results

    Returns
    -------
    fvals:
        List of arrays of function values for each result
    offset_y:
        offset for the y-axis
    """
    min_val = np.inf
    fvals_all = []
    for result in results:
        fvals = np.array(result.optimize_result.fval)

        result_start_indices = process_start_indices(result, start_indices)
        fvals = fvals[result_start_indices]
        # if none of the fvals are finite, set default value to zero as
        # np.nanmin will error for an empty array
        if np.isfinite(fvals).any():
            min_val = min(min_val, np.nanmin(fvals[np.isfinite(fvals)]))

        fvals_all.append(fvals)

    # if there are references, also account for those
    if references:
        min_val = min(min_val, np.nanmin([r["fval"] for r in references]))

    offset_y = process_offset_y(offset_y, scale_y, float(min_val))

    # return offsetted values
    return [fvals + offset_y for fvals in fvals_all], offset_y


def get_ordering_by_start_id(results: Sequence[Result]) -> list[int]:
    """Get an ordering of start IDs.

    The ordering is generated by taking the best function value for each
    start across all results, then sorting these best function values.
    This means that the minimum value of the waterfall plot is
    monotonically increasing.

    Parameters
    ----------
    results:
        The results.

    Returns
    -------
    The ordering.
    """
    if len(results) < 2:
        raise ValueError("Multiple result objects are required.")

    optimize_results = [r.optimize_result.list for r in results]

    # Check whether the same start IDs exist across all results.
    # Note that start IDs, and not the vectors themselves, are checked. This is
    # because, for example, when comparing hierarchical optimization to
    # standard optimization, the starts may be comparable even if the vectors
    # are not identical.
    ids0 = sorted([s.id for s in optimize_results[0]])
    for optimize_result in optimize_results[1:]:
        ids = sorted([s.id for s in optimize_result])
        if ids != ids0:
            raise ValueError("The start IDs of the results do not match.")

    start_fval_pairs = [
        (start.id, start.fval)
        for result in results
        for start in result.optimize_result.list
    ]

    sorted_start_fval_pairs = sorted(
        start_fval_pairs,
        key=lambda pair: pair[1],
    )

    ordering = []
    for start_id, _ in sorted_start_fval_pairs:
        if start_id in ordering:
            continue
        ordering.append(start_id)

    return ordering


def handle_options(ax, max_len_fvals, ref, y_limits, offset_y):
    """
    Apply post-plotting transformations to the axis object.

    Get the limits for the y-axis, plots the reference points, will do
    more at a later time point.

    Parameters
    ----------
    ax: matplotlib.Axes, optional
        Axes object to use.
    max_len_fvals: int
        maximum number of points
    ref: list, optional
        List of reference points for optimization results, containing at
        least a function value fval
    y_limits: float or ndarray, optional
        maximum value to be plotted on the y-axis, or y-limits
    offset_y:
        offset for the y-axis, if it is supposed to be in log10-scale

    Returns
    -------
    ax: matplotlib.Axes
        The plot axes.
    """
    # handle reference points
    for i_ref in ref:
        # plot reference point as line
        ax.plot(
            [0, max_len_fvals - 1],
            [i_ref.fval + offset_y, i_ref.fval + offset_y],
            "--",
            color=i_ref.color,
            label=i_ref.legend,
        )

        # create legend for reference points
        if i_ref.legend is not None:
            ax.legend()

    # handle y-limits
    ax = process_y_limits(ax, y_limits)

    return ax
