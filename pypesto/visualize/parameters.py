import logging
from collections.abc import Callable, Iterable, Sequence

import matplotlib
import matplotlib.axes
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from pypesto.util import delete_nan_inf

from ..C import (
    COLOR,
    INNER_PARAMETERS,
    LOG10,
    WATERFALL_MAX_VALUE,
    InnerParameterType,
)
from ..result import Result
from ._style import resolve_style
from .clust_color import assign_colors
from .misc import (
    _UNSET,
    get_ax,
    get_axes_array,
    plot_density_panel,
    plot_diagonal_marginal,
    process_deprecated_kwarg,
    process_parameter_indices,
    process_result_list,
    process_start_indices,
)
from .reference_points import ReferencePoint, create_references

try:
    from ..hierarchical.base_problem import scale_value
    from ..hierarchical.relative import RelativeInnerProblem
    from ..hierarchical.semiquantitative import SemiquantProblem
except ImportError:
    pass

logger = logging.getLogger(__name__)


def parameters(
    results: Result | Sequence[Result],
    ax: matplotlib.axes.Axes | None = None,
    parameter_indices: str | Sequence[int] = "free_only",
    lb: np.ndarray | list[float] | None = None,
    ub: np.ndarray | list[float] | None = None,
    size: tuple[float, float] | None = None,
    reference: list[ReferencePoint] | None = None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: str | list[str] | None = None,
    balance_alpha: bool = True,
    start_indices: int | Iterable[int] | None = None,
    scale_to_interval: tuple[float, float] | None = None,
    plot_inner_parameters: bool = True,
    log10_scale_hier_sigma: bool = True,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot parameter values.

    Parameters
    ----------
    results:
        Optimization result obtained by 'optimize.py' or list of those
    ax:
        Axes object to use.
    parameter_indices:
        Specifies which parameters should be plotted. Allowed string values are
        'all' (both fixed and free parameters will be plotted)  and
        'free_only' (only free parameters will be plotted)
    lb, ub:
        If not None, override result.problem.lb, problem.problem.ub.
        Dimension either result.problem.dim or result.problem.dim_full.
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified
    reference:
        List of reference points for optimization results, containing at
        least a function value fval
    colors:
        list of colors recognized by matplotlib, or single color
        If not set, clustering is done and colors are assigned automatically
    legends:
        Labels for line plots, one label per result object
    balance_alpha:
        Flag indicating whether alpha for large clusters should be reduced to
        avoid overplotting (default: True)
    start_indices:
        list of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted
    scale_to_interval:
        Tuple of bounds to which to scale all parameter values and bounds, or
        ``None`` to use bounds as determined by ``lb, ub``.
    plot_inner_parameters:
        Flag indicating whether to plot inner parameters (default: True).
    log10_scale_hier_sigma:
        Flag indicating whether to scale inner parameters of type
        ``InnerParameterType.SIGMA`` to log10 (default: True).
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``cmap_discrete``, ``mle_color``, ``outlier_color`` — colours
          of the per-start parameter traces when clustering is used
          (best cluster, secondary clusters, isolated starts respectively).
          Only consulted when ``colors`` is ``None``; an explicit
          ``colors`` short-circuits clustering.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    ax:
        The plot axes.
    """
    style = resolve_style(style_kwargs)

    # parse input
    (results, colors, legends) = process_result_list(
        results, colors, legends, style=style
    )

    if isinstance(parameter_indices, str):
        if parameter_indices == "all":
            parameter_indices = range(0, results[0].problem.dim_full)
        elif parameter_indices == "free_only":
            parameter_indices = results[0].problem.x_free_indices
        else:
            raise ValueError(
                "Permissible values for parameter_indices are "
                "'all', 'free_only' or a list of indices"
            )

    def scale_parameters(x):
        """Scale `x` from [lb, ub] to interval given by `scale_to_interval`."""
        if scale_to_interval is None or scale_to_interval is False:
            return x

        return scale_to_interval[0] + (x - lb) / (ub - lb) * (
            scale_to_interval[1] - scale_to_interval[0]
        )

    for j, result in enumerate(results):
        # handle results and bounds
        (lb, ub, x_labels, fvals, xs, x_axis_label) = handle_inputs(
            result=result,
            lb=lb,
            ub=ub,
            parameter_indices=parameter_indices,
            start_indices=start_indices,
            plot_inner_parameters=plot_inner_parameters,
            log10_scale_hier_sigma=log10_scale_hier_sigma,
        )

        # parse fvals and parameters
        fvals = np.array(fvals)
        # remove nan or inf values
        xs, fvals = delete_nan_inf(
            fvals=fvals,
            x=xs,
            xdim=len(ub) if ub is not None else 1,
            magnitude_bound=WATERFALL_MAX_VALUE,
        )

        lb, ub, xs = map(scale_parameters, (lb, ub, xs))

        # call lowlevel routine
        ax = parameters_lowlevel(
            xs=xs,
            fvals=fvals,
            lb=lb,
            ub=ub,
            x_labels=x_labels,
            x_axis_label=x_axis_label,
            ax=ax,
            size=size,
            colors=colors[j],
            legend_text=legends[j],
            balance_alpha=balance_alpha,
            style=style,
        )

    # parse and apply plotting options
    ref = create_references(references=reference)

    # plot reference points
    for i_ref in ref:
        # reduce parameter vector in reference point, if necessary
        if len(parameter_indices) < results[0].problem.dim_full:
            x_ref = np.array(
                results[0].problem.get_reduced_vector(
                    i_ref["x"], parameter_indices
                )
            )
        else:
            x_ref = np.array(i_ref["x"])
        x_ref = np.reshape(x_ref, (1, x_ref.size))
        x_ref = scale_parameters(x_ref)

        # plot reference parameters using lowlevel routine
        ax = parameters_lowlevel(
            x_ref,
            [i_ref["fval"]],
            ax=ax,
            colors=i_ref["color"],
            linestyle="--",
            legend_text=i_ref.legend,
            balance_alpha=balance_alpha,
        )

    return ax


def parameter_hist(
    result: Result,
    parameter_name: str,
    start_indices: int | list[int] | None = None,
    plot_type: str = "both",
    bins: int | str = "auto",
    bw_method: str = "scott",
    show_bounds: bool = True,
    title: str | None = "Parameter histogram",
    size: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    style_kwargs: dict | None = None,
    color: COLOR = _UNSET,
) -> matplotlib.axes.Axes:
    """
    Plot one parameter's values across starts as a histogram + KDE + rug.

    Parameters
    ----------
    result:
        Optimization result obtained by 'optimize.py'.
    parameter_name:
        Name of the parameter to plot.
    start_indices:
        Which optimization starts to include: a list of indices, or an int
        ``n`` for the first ``n`` starts. Default: all starts.
    plot_type: {'hist'|'kde'|'both'}
        Histogram only, KDE line only, or both with rug marks (default).
    bins:
        Number of bins, or a matplotlib binning strategy (``'auto'``,
        ``'sturges'``, …). Passed to ``ax.hist``.
    bw_method: {'scott', 'silverman' | scalar | pair of scalars}
        Kernel bandwidth method for the KDE overlay.
    show_bounds:
        If ``True`` (default) draw the parameter bound lines and frame the
        x-axis to include them; if ``False`` frame tightly to the data.
    title:
        Axes title. Pass ``None`` to suppress.
    size:
        Figure size in inches. Defaults to matplotlib's default.
    ax:
        Axes object to use.
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
    color:
        Deprecated. Pass ``style_kwargs`` instead — see
        ``rectangle_color`` / ``line_color`` / ``dash_color`` above.

    Returns
    -------
    ax:
        The plot axes.
    """
    process_deprecated_kwarg(
        canonical_name=None,
        canonical_value=None,
        deprecated_name="color",
        deprecated_value=color,
        note=(
            "Pass style_kwargs={'rectangle_color': ..., 'line_color': ..., "
            "'dash_color': ...} instead."
        ),
    )
    style = resolve_style(style_kwargs)
    ax = get_ax(ax, size)

    xs = result.optimize_result.x

    # reduce number of displayed results
    if isinstance(start_indices, int):
        xs = xs[:start_indices]
    elif start_indices is not None:
        xs = [xs[ind] for ind in start_indices]

    parameter_index = result.problem.x_names.index(parameter_name)
    parameter_values = np.array([x[parameter_index] for x in xs])

    # bounds and scale for this parameter
    lb_val = result.problem.lb_full[parameter_index]
    ub_val = result.problem.ub_full[parameter_index]
    x_scales = getattr(result.problem, "x_scales", None)
    scale = x_scales[parameter_index] if x_scales is not None else None

    bound_handle = plot_density_panel(
        ax,
        parameter_values,
        bins=bins,
        bw_method=bw_method,
        style=style,
        show_hist=(plot_type in ("hist", "both")),
        show_kde=(plot_type in ("kde", "both")),
        show_rug=(plot_type in ("hist", "both")),
        show_bounds=show_bounds,
        lb=lb_val,
        ub=ub_val,
    )

    legend_handles, legend_labels = [], []
    show_kde = plot_type in ("kde", "both")
    show_rug = plot_type in ("hist", "both")
    finite_vals = parameter_values[np.isfinite(parameter_values)]
    if finite_vals.size > 0:
        if show_kde:
            legend_handles.append(
                Line2D(
                    [0], [0], color=style["line_color"], lw=style["linewidth"]
                )
            )
            legend_labels.append("KDE")
        if show_rug:
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
            legend_labels.append("Starts")

    if bound_handle is not None:
        legend_handles.append(bound_handle)
        legend_labels.append("Bounds")

    if legend_handles:
        ax.legend(handles=legend_handles, labels=legend_labels)

    xlabel = (
        f"{parameter_name} ({scale})" if scale is not None else parameter_name
    )
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    return ax


def parameters_lowlevel(
    xs: np.ndarray,
    fvals: np.ndarray,
    lb: np.ndarray | list[float] | None = None,
    ub: np.ndarray | list[float] | None = None,
    x_labels: Iterable[str] | None = None,
    x_axis_label: str = "Parameter value",
    ax: matplotlib.axes.Axes | None = None,
    size: tuple[float, float] | None = None,
    colors: Sequence[np.ndarray | COLOR] | None = None,
    linestyle: str = "-",
    legend_text: str | None = None,
    balance_alpha: bool = True,
    style: dict | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot parameters plot using list of parameters.

    Parameters
    ----------
    xs:
        Including optimized parameters for each start that did not result in an infinite fval.
        Shape: (n_starts_successful, dim).
    fvals:
        Function values. Needed to assign cluster colors.
    lb, ub:
        The lower and upper bounds.
    x_labels:
        Labels to be used for the parameters.
    ax:
        Axes object to use.
    size:
        see parameters
    colors:
        A single color recognized by matplotlib or a list of colors, one for each element in 'fvals'.
    linestyle:
        linestyle argument for parameter plot
    legend_text:
        Label for line plots
    balance_alpha:
        Flag indicating whether alpha for large clusters should be reduced to
        avoid overplotting (default: True)
    style:
        Pre-resolved visualization style dict, as returned by
        :func:`pypesto.visualize._style.resolve_style`. When ``None``, defaults
        are used.

    Returns
    -------
    ax:
        The plot axes.
    """
    if size is None:
        # 0.5 inch height per parameter
        size = (18.5, max(xs.shape[1], 1) / 2)

    ax = get_ax(ax, size)

    # assign colors
    colors = assign_colors(
        vals=fvals, colors=colors, balance_alpha=balance_alpha, style=style
    )

    # parameter indices
    parameters_ind = list(range(1, xs.shape[1] + 1))[::-1]

    # plot parameters
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for j_x, x in reversed(list(enumerate(xs))):
        if j_x == 0:
            tmp_legend = legend_text
        else:
            tmp_legend = None
        ax.plot(
            x,
            parameters_ind,
            linestyle,
            color=colors[j_x],
            marker="o",
            label=tmp_legend,
        )

    ax.set_yticks(parameters_ind)
    if x_labels is not None:
        ax.set_yticklabels(x_labels)

    # draw bounds
    parameters_ind = np.array(parameters_ind).flatten()
    if lb is not None:
        lb = np.array(lb, dtype="float64")
        ax.plot(lb.flatten(), parameters_ind, "k--", marker="+")
    if ub is not None:
        ub = np.array(ub, dtype="float64")
        ax.plot(ub.flatten(), parameters_ind, "k--", marker="+")

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Parameter")
    ax.set_title("Estimated parameters")
    if legend_text is not None:
        ax.legend()

    return ax


def handle_inputs(
    result: Result,
    parameter_indices: list[int],
    lb: np.ndarray | list[float] | None = None,
    ub: np.ndarray | list[float] | None = None,
    start_indices: int | Iterable[int] | None = None,
    plot_inner_parameters: bool = False,
    log10_scale_hier_sigma: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[np.ndarray]]:
    """
    Compute the correct bounds for the parameter indices to be plotted.

    Outputs the corresponding parameters and their labels.

    Parameters
    ----------
    result:
        Optimization result obtained by 'optimize.py'.
    parameter_indices:
        Specifies which parameters should be plotted.
    lb, ub:
        If not None, override result.problem.lb, problem.problem.ub.
        Dimension either result.problem.dim or result.problem.dim_full.
    start_indices:
        list of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted
    plot_inner_parameters:
        Flag indicating whether inner parameters should be plotted.
    log10_scale_hier_sigma:
        Flag indicating whether to scale inner parameters of type
        ``InnerParameterType.SIGMA`` to log10 (default: True).

    Returns
    -------
    lb, ub:
        Dimension either result.problem.dim or result.problem.dim_full.
    x_labels:
        ytick labels to be applied later on
    fvals:
        objective function values which are needed for plotting later
    xs:
        parameter values which will be plotted later
    x_axis_label:
        label for the x-axis
    """
    # retrieve results
    fvals = result.optimize_result.fval
    xs = result.optimize_result.x

    # retrieve inner parameters in case of hierarchical optimization
    (
        inner_xs,
        inner_xs_names,
        inner_xs_scales,
        inner_lb,
        inner_ub,
    ) = _handle_inner_inputs(result, log10_scale_hier_sigma)

    # parse indices which should be plotted
    if start_indices is not None:
        start_indices = process_start_indices(result, start_indices)

        # reduce number of displayed results
        xs_out = [xs[ind] for ind in start_indices]
        fvals_out = [fvals[ind] for ind in start_indices]
        if inner_xs is not None and plot_inner_parameters:
            inner_xs_out = [inner_xs[ind] for ind in start_indices]
    else:
        # use non-reduced versions
        xs_out = xs
        fvals_out = fvals
        if inner_xs is not None and plot_inner_parameters:
            inner_xs_out = inner_xs

    # get bounds
    if lb is None:
        lb = result.problem.lb_full
    if ub is None:
        ub = result.problem.ub_full

    # get labels as x_names and scales
    x_labels = list(
        zip(result.problem.x_names, result.problem.x_scales, strict=True)
    )

    # handle fixed and free indices
    if len(parameter_indices) < result.problem.dim_full:
        for ix, x in enumerate(xs_out):
            xs_out[ix] = result.problem.get_reduced_vector(
                x, parameter_indices
            )
        lb = result.problem.get_reduced_vector(lb, parameter_indices)
        ub = result.problem.get_reduced_vector(ub, parameter_indices)
        x_labels = [x_labels[int(i)] for i in parameter_indices]
    else:
        lb = result.problem.lb_full
        ub = result.problem.ub_full

    if inner_xs is not None and plot_inner_parameters:
        lb = np.concatenate([lb, inner_lb])
        ub = np.concatenate([ub, inner_ub])
        inner_xs_labels = list(
            zip(inner_xs_names, inner_xs_scales, strict=True)
        )
        x_labels = x_labels + inner_xs_labels
        xs_out = [
            np.concatenate([x, inner_x]) if x is not None else None
            for x, inner_x in zip(xs_out, inner_xs_out, strict=True)
        ]

    # If all the scales are the same, put it in the x_axis_label
    if len({x_scale for _, x_scale in x_labels}) == 1:
        x_axis_label = "Parameter value (" + x_labels[0][1] + ")"
        x_labels = [x_name for x_name, _ in x_labels]
    else:
        x_axis_label = "Parameter value"
        x_labels = [f"{x_name} ({x_scale})" for x_name, x_scale in x_labels]

    return lb, ub, x_labels, fvals_out, xs_out, x_axis_label


def _handle_inner_inputs(
    result: Result,
    log10_scale_hier_sigma: bool = True,
) -> (
    tuple[None, None, None, None, None]
    | tuple[list[np.ndarray], list[str], list[str], np.ndarray, np.ndarray]
):
    """Handle inner parameters from hierarchical optimization, if available.

    Parameters
    ----------
    result:
        Optimization result obtained by 'optimize.py'.
    log10_scale_hier_sigma:
        Flag indicating whether to scale inner parameters of type
        ``InnerParameterType.SIGMA`` to log10 (default: True).

    Returns
    -------
    inner_xs:
        Inner parameter values which will be appended to xs.
    inner_xs_names:
        Inner parameter names.
    inner_xs_scales:
        Inner parameter scales.
    inner_lb:
        Inner parameter lower bounds.
    inner_ub:
        Inner parameter upper bounds.
    """
    inner_xs = [
        res.get(INNER_PARAMETERS, None) for res in result.optimize_result.list
    ]
    inner_xs_names = None
    inner_xs_scales = None
    inner_lb = None
    inner_ub = None

    from ..problem import HierarchicalProblem

    if any(inner_x is not None for inner_x in inner_xs) and isinstance(
        result.problem, HierarchicalProblem
    ):
        inner_xs_names = result.problem.inner_x_names
        # replace None with a list of nans
        inner_xs = [
            (
                np.full(len(inner_xs_names), np.nan)
                if inner_xs_for_start is None
                else np.asarray(inner_xs_for_start)
            )
            for inner_xs_for_start in inner_xs
        ]
        # set bounds for inner parameters
        inner_lb = result.problem.inner_lb
        inner_ub = result.problem.inner_ub

        # Scale inner parameter bounds according to their parameters scales
        inner_xs_scales = result.problem.inner_scales

        if log10_scale_hier_sigma:
            inner_problems_with_sigma = [
                inner_calculator.inner_problem
                for inner_calculator in result.problem.objective.calculator.inner_calculators
                if isinstance(
                    inner_calculator.inner_problem, RelativeInnerProblem
                )
                or isinstance(inner_calculator.inner_problem, SemiquantProblem)
            ]
            for inner_problem in inner_problems_with_sigma:
                for inner_x_idx, inner_x_name in enumerate(inner_xs_names):
                    if (inner_x_name in inner_problem.get_x_ids()) and (
                        inner_problem.get_for_id(
                            inner_x_name
                        ).inner_parameter_type
                        == InnerParameterType.SIGMA
                    ):
                        # Scale all values, lower and upper bounds
                        for inner_x_for_start in inner_xs:
                            inner_x_for_start[inner_x_idx] = scale_value(
                                inner_x_for_start[inner_x_idx], LOG10
                            )
                        inner_xs_scales[inner_x_idx] = LOG10

        for inner_x_idx, inner_scale in enumerate(inner_xs_scales):
            inner_lb[inner_x_idx] = scale_value(
                inner_lb[inner_x_idx], inner_scale
            )
            inner_ub[inner_x_idx] = scale_value(
                inner_ub[inner_x_idx], inner_scale
            )

    if inner_xs_names is None:
        inner_xs = None

    return inner_xs, inner_xs_names, inner_xs_scales, inner_lb, inner_ub


def parameters_correlation_matrix(
    result: Result,
    parameter_indices: str | Sequence[int] = "free_only",
    start_indices: int | Iterable[int] | None = None,
    method: str | Callable = "pearson",
    cluster: bool = True,
    cmap: Colormap | str = "bwr",
    return_table: bool = False,
    heatmap_kwargs: dict | None = None,
    size: tuple[float, float] | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot correlation of optimized parameters.

    Parameters
    ----------
    result:
        Optimization result obtained by 'optimize.py'
    parameter_indices:
        List of integers specifying the parameters to be considered.
    start_indices:
        List of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted
    method:
        The method to compute correlation. Allowed values are ``pearson``,
        ``kendall``, ``spearman`` or a callable.
    cluster:
        Whether to cluster the correlation matrix.
    cmap:
        Colormap to use for the heatmap. Defaults to 'bwr'.
    return_table:
        Whether to return the parameter table additionally for further
        inspection.
    heatmap_kwargs:
        Additional keyword arguments to :func:`seaborn.heatmap`.
    size:
        Figure size (width, height) in inches.

    Returns
    -------
    ax:
        The plot axis.
    """
    import seaborn as sns

    start_indices = process_start_indices(
        start_indices=start_indices, result=result
    )
    parameter_indices = process_parameter_indices(
        parameter_indices=parameter_indices, result=result
    )
    # put all parameters into a dataframe, where columns are parameters
    parameters = [
        result.optimize_result[i_start]["x"][parameter_indices]
        for i_start in start_indices
    ]
    x_labels = [
        result.problem.x_names[parameter_index]
        for parameter_index in parameter_indices
    ]
    df = pd.DataFrame(parameters, columns=x_labels)
    corr_matrix = df.corr(method=method)
    heatmap_kwargs = {
        "data": corr_matrix,
        "yticklabels": True,
        "vmin": -1,
        "vmax": 1,
        "cmap": cmap,
        "linewidth": 1,
    } | (heatmap_kwargs or {})
    if cluster:
        if size is not None:
            heatmap_kwargs["figsize"] = size
        ax = sns.clustermap(**heatmap_kwargs)
    else:
        ax = sns.heatmap(**heatmap_kwargs)
        if size is not None:
            ax.figure.set_size_inches(*size)
    if return_table:
        return ax, df
    return ax


def optimization_scatter(
    result: Result,
    parameter_indices: str | Sequence[int] = "free_only",
    start_indices: int | Iterable[int] | None = None,
    diag_kind: str = "kde",
    suptitle: str | None = None,
    size: tuple[float, float] | None = None,
    show_bounds: bool = False,
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Plot a scatter matrix of all parameter pairs for the given starts.

    Parameters
    ----------
    result:
        Optimization result obtained by ‘optimize.py’.
    parameter_indices:
        List of integers specifying the parameters to be considered.
    start_indices:
        List of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted.
    diag_kind:
        Marginal distribution shown on the diagonal: ``’kde’`` (default)
        or ``’hist’``.
    suptitle:
        Title of the figure.
    size:
        Figure size (width, height) in inches. Defaults to
        ``(2.5 * n + 0.5, 2.5 * n + 0.5)``.
    show_bounds:
        Whether to draw dashed lines at the parameter bounds.
    axes:
        Optional axes grid to draw into. Must have shape
        ``(n_params, n_params)``.

    Returns
    -------
    axes:
        2-D NumPy array of shape ``(n_params, n_params)`` containing one
        matplotlib Axes per panel.
    """
    import matplotlib.cm as mpl_cm
    from matplotlib.colors import Normalize

    start_indices = process_start_indices(
        start_indices=start_indices, result=result
    )
    parameter_indices = process_parameter_indices(
        parameter_indices=parameter_indices, result=result
    )

    n = len(parameter_indices)
    x_labels = [result.problem.x_names[i] for i in parameter_indices]

    # data matrix: rows = starts, cols = selected parameters
    data = np.array(
        [
            result.optimize_result[i]["x"][parameter_indices]
            for i in start_indices
        ]
    )
    fvals = np.array([result.optimize_result[i].fval for i in start_indices])

    # continuous colormap: viridis, low fval (best) → yellow, high fval (worst) → dark
    cmap = matplotlib.colormaps["viridis_r"]
    min_fval_range = 1.0
    fval_min = fvals.min()
    fval_max = fvals.max()
    fval_mid = 0.5 * (fval_min + fval_max)
    fval_half_range = max(fval_max - fval_min, min_fval_range) / 2
    fval_norm = Normalize(
        vmin=fval_mid - fval_half_range,
        vmax=fval_mid + fval_half_range,
    )

    if size is None and axes is None:
        size = (2.5 * n + 0.5, 2.5 * n + 0.5)

    axes = get_axes_array(axes=axes, nrows=n, ncols=n, size=size)
    fig = axes.flat[0].figure
    fig.set_layout_engine("constrained")

    previous_colorbar_axes = []
    for ax in axes.flat:
        colorbar_ax = getattr(
            ax,
            "_pypesto_optimization_scatter_colorbar_ax",
            None,
        )
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
        if hasattr(ax, "_pypesto_optimization_scatter_colorbar_ax"):
            delattr(ax, "_pypesto_optimization_scatter_colorbar_ax")

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            col_vals = data[:, col]
            row_vals = data[:, row]

            if row == col:
                plot_diagonal_marginal(
                    ax=ax, values=col_vals, diag_kind=diag_kind
                )
            else:
                ax.scatter(
                    col_vals,
                    row_vals,
                    c=fvals,
                    cmap=cmap,
                    norm=fval_norm,
                    s=35,
                    alpha=0.85,
                    linewidths=0.6,
                    edgecolors="white",
                    zorder=3,
                )
                ax.set_ylabel(x_labels[row])

            ax.set_xlabel(x_labels[col])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if show_bounds:
                pi_col = parameter_indices[col]
                pi_row = parameter_indices[row]
                for val in (
                    result.problem.lb_full[pi_col],
                    result.problem.ub_full[pi_col],
                ):
                    ax.axvline(val, color="k", ls="--", lw=0.8)
                if row != col:
                    for val in (
                        result.problem.lb_full[pi_row],
                        result.problem.ub_full[pi_row],
                    ):
                        ax.axhline(val, color="k", ls="--", lw=0.8)

    # shared x-limits per column, shared y-limits per row (non-diagonal)
    for col in range(n):
        vals = data[:, col]
        data_range = vals.max() - vals.min()
        pad = data_range * 0.1 if data_range > 0 else 0.5
        xlim = (vals.min() - pad, vals.max() + pad)
        for row in range(n):
            axes[row, col].set_xlim(xlim)
    for row in range(n):
        vals = data[:, row]
        data_range = vals.max() - vals.min()
        pad = data_range * 0.1 if data_range > 0 else 0.5
        ylim = (vals.min() - pad, vals.max() + pad)
        for col in range(n):
            if col != row:
                axes[row, col].set_ylim(ylim)

    sm = mpl_cm.ScalarMappable(cmap=cmap, norm=fval_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, pad=0.03)
    cbar.set_label("Objective value")
    for ax in axes.flat:
        ax._pypesto_optimization_scatter_colorbar_ax = cbar.ax

    if suptitle:
        fig.suptitle(suptitle)

    return axes
