import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.ticker import MaxNLocator
import numpy as np

from typing import Iterable, List, Optional, Sequence, Tuple, Union

from ..result import Result
from .reference_points import create_references, ReferencePoint
from .clust_color import assign_colors
from .clust_color import delete_nan_inf
from .misc import process_result_list, process_start_indices


def parameters(
        results: Union[Result, Sequence[Result]],
        ax: Optional[matplotlib.axes.Axes] = None,
        parameter_indices: Union[str, Sequence[int]] = 'free_only',
        lb: Optional[Union[np.ndarray, List[float]]] = None,
        ub: Optional[Union[np.ndarray, List[float]]] = None,
        size: Optional[Tuple[float, float]] = None,
        reference: Optional[List[ReferencePoint]] = None,
        colors: Optional[Union[List[float], List[List[float]]]] = None,
        legends: Optional[Union[str, List[str]]] = None,
        balance_alpha: bool = True,
        start_indices: Optional[Union[int, Iterable[int]]] = None
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
        list of RGBA colors, or single RGBA color
        If not set, clustering is done and colors are assigned automatically
    legends:
        Labels for line plots, one label per result object
    balance_alpha:
        Flag indicating whether alpha for large clusters should be reduced to
        avoid overplotting (default: True)
    start_indices:
        list of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted

    Returns
    -------
    ax:
        The plot axes.
    """

    # parse input
    (results, colors, legends) = process_result_list(results, colors, legends)

    if isinstance(parameter_indices, str):
        if parameter_indices == 'all':
            parameter_indices = range(0, results[0].problem.dim_full)
        elif parameter_indices == 'free_only':
            parameter_indices = results[0].problem.x_free_indices
        else:
            raise ValueError("Permissible values for parameter_indices are "
                             "'all', 'free_only' or a list of indices")

    for j, result in enumerate(results):
        # handle results and bounds
        (lb, ub, x_labels, fvals, xs) = \
            handle_inputs(result=result, lb=lb, ub=ub,
                          parameter_indices=parameter_indices,
                          start_indices=start_indices)

        # call lowlevel routine
        ax = parameters_lowlevel(xs=xs, fvals=fvals, lb=lb, ub=ub,
                                 x_labels=x_labels, ax=ax, size=size,
                                 colors=colors[j], legend_text=legends[j],
                                 balance_alpha=balance_alpha)

    # parse and apply plotting options
    ref = create_references(references=reference)

    # plot reference points
    for i_ref in ref:
        # reduce parameter vector in reference point, if necessary
        if len(parameter_indices) < results[0].problem.dim_full:
            x_ref = np.array(results[0].problem.get_reduced_vector(
                i_ref['x'],
                parameter_indices))
        else:
            x_ref = np.array(i_ref['x'])
        x_ref = np.reshape(x_ref, (1, x_ref.size))

        # plot reference parameters using lowlevel routine
        ax = parameters_lowlevel(x_ref, [i_ref['fval']], ax=ax,
                                 colors=i_ref['color'],
                                 linestyle='--',
                                 legend_text=i_ref.legend,
                                 balance_alpha=balance_alpha)

    return ax


def parameter_hist(
        result: Result,
        parameter_name: str,
        bins: Union[int, str] = 'auto',
        ax: Optional['matplotlib.Axes'] = None,
        size: Optional[Tuple[float]] = (18.5, 10.5),
        color: Optional[List[float]] = None,
        start_indices: Optional[Union[int, List[int]]] = None):
    """
    Plot parameter values as a histogram.

    Parameters
    ----------
    result:
        Optimization result obtained by 'optimize.py'
    parameter_name:
        The name of the parameter that should be plotted
    bins:
        Specifies bins of the histogram
    ax:
        Axes object to use
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified
    color:
        RGBA color.
    start_indices:
        List of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted


    Returns
    -------
    ax:
    The plot axes.

    """

    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    xs = result.optimize_result.get_for_key('x')

    # reduce number of displayed results
    if isinstance(start_indices, int):
        xs = xs[:start_indices]
    elif start_indices is not None:
        xs = [xs[ind] for ind in start_indices]

    parameter_index = result.problem.x_names.index(parameter_name)
    parameter_values = [x[parameter_index] for x in xs]

    ax.hist(parameter_values, color=color, bins=bins)
    ax.set_xlabel(parameter_name)
    ax.set_ylabel("counts")
    ax.set_title(f"Parameter {parameter_name}")

    return ax


def parameters_lowlevel(
        xs: Sequence[Union[np.ndarray, List[float]]],
        fvals: Union[np.ndarray, List[float]],
        lb: Optional[Union[np.ndarray, List[float]]] = None,
        ub: Optional[Union[np.ndarray, List[float]]] = None,
        x_labels: Optional[Iterable[str]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        size: Optional[Tuple[float, float]] = None,
        colors: Optional[Sequence[Union[np.ndarray, List[float]]]] = None,
        linestyle: str = '-',
        legend_text: Optional[str] = None,
        balance_alpha: bool = True
) -> matplotlib.axes.Axes:

    """
    Plot parameters plot using list of parameters.

    Parameters
    ----------

    xs:
        Including optimized parameters for each startpoint.
        Shape: (n_starts, dim).

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
        One for each element in 'fvals'.

    linestyle:
        linestyle argument for parameter plot

    legend_text:
        Label for line plots

    balance_alpha:
        Flag indicating whether alpha for large clusters should be reduced to
        avoid overplotting (default: True)

    Returns
    -------

    ax:
        The plot axes.
    """

    # parse input
    xs = np.array(xs)
    fvals = np.array(fvals)
    # remove nan or inf values in fvals and xs
    xs, fvals = delete_nan_inf(fvals, xs, len(ub) if ub is not None else 1)

    if size is None:
        # 0.5 inch height per parameter
        size = (18.5, max(xs.shape[1], 1) / 2)

    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # assign colors
    colors = assign_colors(vals=fvals, colors=colors,
                           balance_alpha=balance_alpha)

    # parameter indices
    parameters_ind = list(range(1, xs.shape[1] + 1))[::-1]

    # plot parameters
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for j_x, x in reversed(list(enumerate(xs))):
        if j_x == 0:
            tmp_legend = legend_text
        else:
            tmp_legend = None
        ax.plot(x, parameters_ind,
                linestyle,
                color=colors[j_x],
                marker='o',
                label=tmp_legend)

    plt.yticks(parameters_ind, x_labels)

    # draw bounds
    parameters_ind = np.array(parameters_ind).flatten()
    if lb is not None:
        ax.plot(lb.flatten(), parameters_ind, 'k--', marker='+')
    if ub is not None:
        ax.plot(ub.flatten(), parameters_ind, 'k--', marker='+')

    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Parameter')
    ax.set_title('Estimated parameters')
    if legend_text is not None:
        ax.legend()

    return ax


def handle_inputs(
        result: Result, parameter_indices: List[int],
        lb: Optional[Union[np.ndarray, List[float]]] = None,
        ub: Optional[Union[np.ndarray, List[float]]] = None,
        start_indices: Optional[Union[int, Iterable[int]]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[np.ndarray]]:
    """
    Computes the correct bounds for the parameter indices to be plotted and
    outputs the corresponding parameters and their labels

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
    """

    # retrieve results
    fvals = result.optimize_result.get_for_key('fval')
    xs = result.optimize_result.get_for_key('x')

    # parse indices which should be plotted
    if start_indices is not None:
        start_indices = process_start_indices(start_indices, len(fvals))

        # reduce number of displayed results
        xs_out = [xs[ind] for ind in start_indices]
        fvals_out = [fvals[ind] for ind in start_indices]
    else:
        # use non-reduced versions
        xs_out = xs
        fvals_out = fvals

    # get bounds
    if lb is None:
        lb = result.problem.lb_full
    if ub is None:
        ub = result.problem.ub_full

    # get labels
    x_labels = result.problem.x_names

    # handle fixed and free indices
    if len(parameter_indices) < result.problem.dim_full:
        for ix, x in enumerate(xs_out):
            xs_out[ix] = result.problem.get_reduced_vector(x,
                                                           parameter_indices)
        lb = result.problem.get_reduced_vector(lb, parameter_indices)
        ub = result.problem.get_reduced_vector(ub, parameter_indices)
        x_labels = [x_labels[int(i)] for i in parameter_indices]
    else:
        lb = result.problem.get_full_vector(lb)
        ub = result.problem.get_full_vector(ub)

    return lb, ub, x_labels, fvals_out, xs_out
