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
    pre_col = scalarmap.to_rgba(range(vmax+1))

    # grey color for 1-size cluster
    pre_col[vmax] = (0.7,0.7,0.7,1)
    ind_col = np.zeros(len(clust))
    sum_col = 0
    for iind, value_ind in enumerate(ind_clust):
        if clustsize[value_ind] > 1:
            ind_col[iind] = int(sum_col)
            if iind < len(ind_clust)-1:
                if value_ind != ind_clust[iind+1]:
                    sum_col = sum_col + 1
        else:
            ind_col[iind] = vmax
            
    # indices for colors
    ind_col = [int(ind) for ind in ind_col]
    col = pre_col[ind_col]

    return col
