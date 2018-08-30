from scipy import cluster
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np


def get_clust(result_fval):

    """
    Cluster cost function values

    Parameters
    ----------

    result_fval: numeric list or array
        Including values need to be plotted.

    Returns
    -------

    clust: numeric list
         Indicating the corresponding cluster of each element from 'result_fval'.

    clustsize: numeric list
        size of clusters, form 1 to number of clusters.

    ind_clust: numeric list
        Indices to reconstruct 'clust' from a list with 1:number of clusters
    """

    clust = cluster.hierarchy.fcluster(
        cluster.hierarchy.linkage(result_fval),
        0.1, criterion='distance')
    uclust, ind_clust = np.unique(clust, return_inverse=True)
    clustsize = np.zeros(len(uclust))
    for iclustsize, value_uclust in enumerate(uclust):
        clustsize[iclustsize] = sum(clust == value_uclust)

    return clust, clustsize, ind_clust


def assigncolor(result_fval):

    """
    Assign color to each cluster

    Parameters
    ----------

    result_fval: numeric list or array
        Including values need to be plotted.

    Returns
    -------

    Col: list of RGB
        One for each element in 'result_fval'
    """

    clust, clustsize, ind_clust = get_clust(result_fval)
    vmax = max(clust) - sum(clustsize == 1)
    cnorm = colors.Normalize(vmin=0, vmax=vmax)
    scalarmap = cm.ScalarMappable(norm=cnorm)
    uind_col = vmax * np.ones(len(clustsize))
    sum_col = 0
    for iclustsize, value_clustsize in enumerate(clustsize):
        if value_clustsize > 1:
            uind_col[iclustsize] = sum_col
            sum_col = sum_col + 1

    ind_col = uind_col[ind_clust]
    col = scalarmap.to_rgba(ind_col)

    return col
