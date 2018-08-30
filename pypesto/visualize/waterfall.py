import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .clust_color import assigncolor


def waterfall(result, ax=None):

    """
    Plot waterfall plot.

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
    return waterfall_lowlevel(result_fval, ax)


def waterfall_lowlevel(result_fval, ax=None):

    """
    Plot waterfall plot using list of cost function values.

    Parameters
    ----------

    result_fval: list of cost function values

    ax: matplotlib.Axes, optional
        Axes object to use.

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # axes
    if ax is None:
        fig, ax = plt.subplots()

    # extract cost function values from result

    result_fval = np.reshape(result_fval, [len(result_fval), 1])
    startpoints = range(1, len(result_fval) + 1)

    # assign color
    col = assigncolor(result_fval)

    # plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(startpoints, result_fval)
    for ifval, fval in enumerate(result_fval):
        ax.plot(ifval+1, fval,
                color=col[ifval], marker='o')

    ax.set_xlabel('Ordered optimizer run')
    ax.set_ylabel('Function value')

    return ax
