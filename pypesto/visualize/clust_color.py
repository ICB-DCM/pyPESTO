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
    _, ind_clust = np.unique(clust, return_index=True)
    uclust = clust[np.sort(list(ind_clust))]
    clustsize = np.zeros(len(uclust))
    for iclustsize, value_uclust in enumerate(uclust):
        clustsize[iclustsize] = sum(clust == value_uclust)

    return clust, clustsize


def assign_clustered_colors(vals):
    """
    Cluster and assign colors.

    Parameters
    ----------

    vals: numeric list or array
        List to be clustered and assigned colors.

    Returns
    -------

    Col: list of RGBA
        One for each element in 'vals'.
    """

    # sanity checks
    if vals is None or len(vals) == 0:
        return []

    # assign clusters
    clust, clustsize = assign_clusters(vals)

    # assign colors
    vmax = max(clust) - sum(clustsize == 1)
    cnorm = colors.Normalize(vmin=0, vmax=vmax)
    scalarmap = cm.ScalarMappable(norm=cnorm)

    # colors for each cluster and one-size clusters
    cols_clust = scalarmap.to_rgba(range(vmax + 1))

    # grey color for 1-size clusters as the last color in 'cols_clust'
    cols_clust[vmax] = (0.7, 0.7, 0.7, 1)

    # find non-1-size clusters in 'clustsize'
    ind_one = np.where(clustsize == 1)[0]
    n_clustsize = np.delete(clustsize, ind_one)

    for icluster in range(vmax+1):
        if icluster == vmax:
            # grey color for 1-size clusters as the last color in 'cols_clust'
            # alpha set according to number of 1-size clusters (+1 to avoid
            # zero division)
            cols_clust[icluster][3] = min(1, 5 / (sum(clustsize == 1) + 1))
        else:
            # normalize alpha according to clustersize 
            cols_clust[icluster][3] = min(1,
                                          5 / n_clustsize[icluster])

    # pre-array of indices for colors
    ind_col = np.zeros(len(clust))

    # number of colors assigned to cluster
    sum_col = 0

    # assign color indices for each point
    i_ind_col = 0
    for val_clustsize in clustsize:
        # if cluster size > 1
        if val_clustsize > 1:
            # assign a color with is not grey
            ind_col[int(i_ind_col):int(i_ind_col + val_clustsize)] = sum_col
            i_ind_col = i_ind_col + val_clustsize
            sum_col = sum_col + 1
        # if cluster size = 1
        else:
            # use grey
            ind_col[int(i_ind_col)] = vmax
            i_ind_col = i_ind_col + 1


    # indices for colors
    ind_col = [int(ind) for ind in ind_col]
    col = cols_clust[ind_col]

    return col
