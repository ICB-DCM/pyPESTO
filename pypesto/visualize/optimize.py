import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from scipy import cluster
import numpy as np


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

    # axes
    if ax is None:
        fig, ax = plt.subplots()

    # extract cost function values from result
    result_fval = result.optimize_result.get_for_key('fval')
    result_fval = np.reshape(result_fval, [len(result_fval), 1])
    startpoints = range(1, len(result_fval) + 1)

    # cluster
    clust = cluster.hierarchy.fcluster(
        cluster.hierarchy.linkage(result_fval),
        0.1, criterion='distance')
    uclust, ind_clust = np.unique(clust, return_inverse=True)
    clustsize = np.zeros(len(uclust))
    for iclustsize in range(len(uclust)):
        clustsize[iclustsize] = sum(clust == uclust[iclustsize])

    # assign colors
    jet = plt.get_cmap('jet')
    vmax = len(uclust) - sum(clustsize == 1)
    cNorm = colors.Normalize(vmin=0, vmax=vmax)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    # plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(startpoints, result_fval)
    sum_grey = 1
    for iind in range(len(ind_clust)):
        if clustsize[ind_clust[iind]] == 1:
            ax.plot(startpoints[iind], result_fval[iind],
                    color='lightgrey', marker='o')
            sum_grey = sum_grey
        else:
            Col = scalarMap.to_rgba(uclust[ind_clust[iind]] - sum_grey)
            ax.plot(startpoints[iind], result_fval[iind],
                    color=Col, marker='o')

    ax.set_xlabel('Ordered multi-starts')
    ax.set_ylabel('Function value')

    return ax
