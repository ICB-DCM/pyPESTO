import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .clust_color import assigncolor


def parameters(result, ax=None):

    """
    Plot parameter values.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'

    ax: matplotlib.Axes, optional
        Axes object to use.

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    result_fval = result.optimize_result.get_for_key('fval')
    result_x = result.optimize_result.get_for_key('x')
    lb = result.problem.lb
    ub = result.problem.ub

    return parameters_lowlevel(result_x, result_fval, lb, ub, ax,)


def parameters_lowlevel(result_x, result_fval, lb, ub, ax=None):

    """
    Plot waterfall plot using list of cost function values.

    Parameters
    ----------

    result_x: nested list or array
        Including optimized parameters for each startpoint.

    result_fval: numeric list or array
        Including values need to be plotted.

    lb, ub: array_like
        The lower and upper bounds. For unbounded problems set to inf.

    ax: matplotlib.Axes, optional
        Axes object to use.

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    if ax is None:
        ax = plt.subplots()[1]

    result_fval = np.reshape(result_fval, [len(result_fval), 1])

    # assign color
    col = assigncolor(result_fval)

    # parameter indices
    parameters_ind = range(1, len(result_x[0]) + 1)

    # plot parameters
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ix, value_x in enumerate(result_x):
        ax.plot(value_x, parameters_ind, color=col[ix], marker='o')

    # draw bounds
    ax.plot(lb[0], parameters_ind, 'b--')
    ax.plot(ub[0], parameters_ind, 'b--')

    ax.set_xlabel('Parameter value')
    ax.set_title('Estimated parameters')

    return ax
