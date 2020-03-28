import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import numbers
from .reference_points import create_references
from .clust_color import assign_colors
from .clust_color import delete_nan_inf
from .misc import process_result_list


def parameters(results, ax=None, free_indices_only=True, lb=None, ub=None,
               size=None, reference=None, colors=None, legends=None,
               balance_alpha=True, start_indices=None):
    """
    Plot parameter values.

    Parameters
    ----------

    results: pypesto.Result or list
        Optimization result obtained by 'optimize.py' or list of those

    ax: matplotlib.Axes, optional
        Axes object to use.

    free_indices_only: bool, optional
        If True, only free parameters are shown. If
        False, also the fixed parameters are shown.

    lb, ub: ndarray, optional
        If not None, override result.problem.lb, problem.problem.ub.
        Dimension either result.problem.dim or result.problem.dim_full.

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    reference: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    colors: list, or RGBA, optional
        list of colors, or single color
        color or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically

    legends: list or str
        Labels for line plots, one label per result object

    balance_alpha: bool (optional)
        Flag indicating whether alpha for large clusters should be reduced to
        avoid overplotting (default: True)

    start_indices: list or int
        list of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # parse input
    (results, colors, legends) = process_result_list(results, colors, legends)

    for j, result in enumerate(results):
        # handle results and bounds
        (lb, ub, x_labels, fvals, xs) = \
            handle_inputs(result=result, lb=lb, ub=ub,
                          free_indices_only=free_indices_only,
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
        if free_indices_only:
            x_ref = np.array(result.problem.get_reduced_vector(i_ref['x']))
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


def parameters_lowlevel(xs, fvals, lb=None, ub=None, x_labels=None,
                        ax=None, size=None, colors=None, linestyle='-',
                        legend_text=None, balance_alpha=True):

    """
    Plot parameters plot using list of parameters.

    Parameters
    ----------

    xs: nested list or array
        Including optimized parameters for each startpoint.
        Shape: (n_starts, dim).

    fvals: numeric list or array
        Function values. Needed to assign cluster colors.

    lb, ub: array_like, optional
        The lower and upper bounds.

    x_labels: array_like of str, optional
        Labels to be used for the parameters.

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        see parameters

    colors: list of RGBA
        One for each element in 'fvals'.

    linestyle: str, optional
        linestyle argument for parameter plot

    legend_text: str
        Label for line plots

    balance_alpha: bool (optional)
        Flag indicating whether alpha for large clusters should be reduced to
        avoid overplotting (default: True)

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # parse input
    xs = np.array(xs)
    fvals = np.array(fvals)
    # remove nan or inf values in fvals and xs
    xs, fvals = delete_nan_inf(fvals, xs)

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
    ax.set_ylabel('Parameter index')
    ax.set_title('Estimated parameters')
    if legend_text is not None:
        ax.legend()

    return ax


def handle_inputs(result, free_indices_only, lb=None, ub=None,
                  start_indices=None):
    """
    Computes the correct bounds for the parameter indices to be plotted and
    outputs the corrsponding parameters and their labels

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'.

    free_indices_only: bool, optional
        If True, only free parameters are shown. If
        False, also the fixed parameters are shown.

    lb, ub: ndarray, optional
        If not None, override result.problem.lb, problem.problem.ub.
        Dimension either result.problem.dim or result.problem.dim_full.

    start_indices: list or int
        list of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted

    Returns
    -------

    lb, ub: ndarray
        Dimension either result.problem.dim or result.problem.dim_full.

    x_labels: list of str
        ytick labels to be applied later on

    fvals: ndarray
        objective function values which are needed for plotting later

    xs: ndarray
        parameter values which will be plotted later
    """

    # retrieve results
    fvals = result.optimize_result.get_for_key('fval')
    xs = result.optimize_result.get_for_key('x')

    # parse indices which should be plotted
    if start_indices is not None:
        # handle, if only a number was passed
        if isinstance(start_indices, numbers.Number):
            start_indices = range(int(start_indices))

        start_indices = np.array(start_indices, dtype=int)

        # reduce number of displayed results
        xs_out = [xs[ind] for ind in start_indices]
        fvals_out = [fvals[ind] for ind in start_indices]
    else:
        # use non-reduced versions
        xs_out = xs
        fvals_out = fvals

    # get bounds
    if lb is None:
        lb = result.problem.lb
    if ub is None:
        ub = result.problem.ub

    # get labels
    x_labels = result.problem.x_names

    # handle fixed and free indices
    if free_indices_only:
        for ix, x in enumerate(xs_out):
            xs_out[ix] = result.problem.get_reduced_vector(x)
        lb = result.problem.get_reduced_vector(lb)
        ub = result.problem.get_reduced_vector(ub)
        x_labels = [x_labels[int(i)] for i in result.problem.x_free_indices]
    else:
        lb = result.problem.get_full_vector(lb)
        ub = result.problem.get_full_vector(ub)

    return lb, ub, x_labels, fvals_out, xs_out
