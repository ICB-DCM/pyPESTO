from scipy import cluster
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
    elif len(vals) == 1:
        return np.array([1]), np.array([1.]), np.array([0])

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

    colors: list of RGBA
        One for each element in 'vals'.
    """

    # sanity checks
    if vals is None or len(vals) == 0:
        return []

    # assign clusters
    clust, cluster_size, cluster_indices = assign_clusters(vals)

    # create list of colors, which has the correct shape
    n_clusters = max(clust) - sum(cluster_size == 1)

    # fill color array from colormap
    colormap = cm.ScalarMappable().to_rgba
    color_list = colormap(np.linspace(0., 1., n_clusters))

    # We have clustered the results. However, clusters may have size 1,
    # so we need to rearrange cluster_indices in order to map all
    # 1-element-clusters to the cluster with index n_clusters

    # create a dummy variable with the colors
    colors = np.zeros((clust.size, 4))

    # run through the cluster_indices and collect which element
    # is in 1-element-clusters, and which is in real clusters
    no_clusters = np.where(cluster_size == 1)[0]
    real_clusters = np.unique(np.where(cluster_size > 1)[0])

    # assign colors to real clusters
    for icol, iclust in enumerate(real_clusters):
        ind_of_iclust = np.argwhere(cluster_indices == iclust).flatten()
        colors[ind_of_iclust, :] = color_list[icol, :]

    # assign color to non-clustered indices
    for noclust in no_clusters:
        ind_of_noclust = np.argwhere(noclust in no_clusters).flatten()
        colors[ind_of_noclust, :] = [0.7, 0.7, 0.7, 1]

    return colors


def assign_colors(vals, colors=None):
    """
    Assign colors or format user specified colors.

    Parameters
    ----------

    vals: numeric list or array
        List to be clustered and assigned colors.

    colors: list, or RGBA, optional
        list of colors, or single color

    Returns
    -------

    colors: list of RGBA
        One for each element in 'vals'.
    """

    # sanity checks
    if vals is None or len(vals) == 0:
        return np.array([])

    # if the user did not specify any colors:
    if colors is None:
        return assign_clustered_colors(vals)

    # The user passed values and colors: parse them first!
    # we want everything to be numpy arrays, to not check everytime wther a
    # list was passed or an ndarray
    colors = np.array(colors)

    # Get number of elements and use user assigned colors
    n_vals = len(vals) if isinstance(vals, list) else vals.size

    # Two usages are possible: One color for the whole data set, or one
    # color for each value:
    if colors.size == 4:
        # Only one color was passed: flatten in case and repeat n_vals times
        if colors.ndim == 2:
            colors = colors[0]
        return np.array([colors] * n_vals)
    else:
        if colors.shape[1] == 4 and n_vals == colors.shape[0]:
            return colors
        elif colors.shape[0] == 4:
            colors = np.transpose(colors)
            if n_vals == colors.shape[0]:
                return colors

        # Shape of array did not match n_vals. Error due to size mismatch:
        raise ('Incorrect color input. Colors must be specified either as '
               'list of [r, g, b, alpha] with length equal to function '
               'values Number of function (here: ' + str(n_vals) + '), or as '
               'one single [r, g, b, alpha] color.')


def assign_colors_for_result_list(num_results, colors=None):
    """
    Creates a list of colors for a list of pypesto.Result objects or checks
    a user-provided list of colors and uses this if everything is ok

    Parameters
    ----------

    num_results: int
        number of results in list

    colors: list, or RGBA, optional
        list of colors, or single color

    Returns
    -------

    colors: list of RGBA
        One for each element in 'vals'.
    """

    # if the user did not specify any colors:
    if colors is None:
        dummy_clusters = np.array(list(range(num_results)) * 2)
        colors = assign_colors(dummy_clusters)
        real_indices = list(range(int(colors.shape[0] / 2)))
        return colors[real_indices]

    # if the user specified color lies does not match the number of results
    if len(colors) != num_results:
        raise ('Incorrect color input. Colors must be specified either as '
               'list of [r, g, b, alpha] with length equal to function '
               'values Number of function (here: ' + str(num_results) + '), '
               'or as one single [r, g, b, alpha] color.')

    return colors
