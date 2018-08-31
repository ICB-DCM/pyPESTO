from scipy import cluster
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np


def assign_clusters(vals):
    """
    Find clustering.

    Parameters
    ----------

    vals: numeric list or array
        List to be clustered.

    Returns
    -------

    clust: numeric list
         Indicating the corresponding cluster of each element from
         'vals'.

    clustsize: numeric list
        Size of clusters, length equals number of clusters.

    ind_clust: numeric list
        Indices to reconstruct 'clust' from a list with 1:number of clusters.
    """

    # sanity checks
    if vals is None or len(vals) == 0:
        return [], [], []

    # linkage requires (n, 1) data array
    vals = np.reshape(vals, (-1, 1))

    clust = cluster.hierarchy.fcluster(
        cluster.hierarchy.linkage(vals),
        0.1, criterion='distance')
    uclust, ind_clust = np.unique(clust, return_inverse=True)
    clustsize = np.zeros(len(uclust))
    for iclustsize, value_uclust in enumerate(uclust):
        clustsize[iclustsize] = sum(clust == value_uclust)

    return clust, clustsize, ind_clust


def assign_clustered_colors(vals):
    """
    Cluster and assign colors.

    Parameters
    ----------

    vals: numeric list or array
        List to be clustered and assigned colors.

    Returns
    -------

    Col: list of RGB
        One for each element in 'vals'.
    """

    # sanity checks
    if vals is None or len(vals) == 0:
        return []

    # assign clusters
    clust, clustsize, ind_clust = assign_clusters(vals)

    # assign colors
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
