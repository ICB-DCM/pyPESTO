from scipy import cluster
import matplotlib.colors as plt_colors
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
    cnorm = plt_colors.Normalize(vmin=0, vmax=vmax)
    scalarmap = cm.ScalarMappable(norm=cnorm)

    # colors for each cluster and one-size clusters
    cols_clust = scalarmap.to_rgba(range(vmax + 1))

    # grey color for 1-size clusters as the last color in 'cols_clust'
    cols_clust[vmax] = (0.7, 0.7, 0.7, 1)

    # pre-array of indices for colors
    ind_col = np.zeros(len(ind_clust))

    # number of colors assigned to cluster
    sum_col = 0

    # assign color indices for each point
    for iind, value_ind in enumerate(ind_clust):
        # if cluster size > 1
        if clustsize[value_ind] > 1:
            # assign a color with is not grey
            ind_col[iind] = sum_col
            # if element is not the last one in 'ind_col'
            if iind < len(ind_clust) - 1:
                # if the next element does not belongs to the seem cluster
                if value_ind != ind_clust[iind + 1]:
                    # use the next color
                    sum_col = sum_col + 1
        # if cluster size = 1
        else:
            # use grey
            ind_col[iind] = vmax

    # indices for colors
    ind_col = [int(ind) for ind in ind_col]
    col = cols_clust[ind_col]

    return col


def assign_colors(vals, colors=None):
    """
    Assign colors or format user specified colors.

    Parameters
    ----------

    vals: numeric list or array
        List to be clustered and assigned colors.

    colors: list, or RGB, optional
        list of colors, or single color

    Returns
    -------

    Col: list of RGB
        One for each element in 'vals'.
    """

    # sanity checks
    if vals is None or len(vals) == 0:
        return []

    # if the user did not specify any colors:
    if colors is None:
        return assign_clustered_colors(vals)

    # get number of elements and use user assigned colors
    n_vals = len(vals) if isinstance(vals, list) else vals.size

    # check correct length
    if any(isinstance(i_color, list) for i_color in colors):
        if len(colors) == n_vals:
            return colors
    else:
        if isinstance(colors, list) and len(colors) == 4:
            return [colors] * n_vals

    raise ('Incorrect color input. Colors must be specified either as '
           'list of [r, g, b, alpha] with length equal to function '
           'values Number of function (here: ' + str(n_vals) + '), or as '
           'one single [r, g, b, alpha] color.')


def handle_result_list(results, colors=None):
    """
    assigns colors to a list of results

    Parameters
    ----------

    results: list or pypesto.Result
        list of pypesto.Result objects or a single pypesto.Result

    colors: list, optional
        list of RGB colors

    Returns
    -------

    results: pypesto.result or list
       list of pypesto.result objects

    colors: list of RGB
        One for each element in 'results'.
    """

    # check if list
    if not isinstance(results, list):
        results = [results]
    else:
        # if more than one result is passed, one color per result is used
        if colors is None:
            colors = assign_colors(range(len(results)), colors)

        # if the user passed a list of colors, does it have the correct length?
        if any(isinstance(i_color, list) for i_color in colors):
            if len(colors) != len(colors):
                raise ('List of results and list of colors is passed. The '
                       'length of the color list must match he length of the '
                       'results list. Stopping.')

    return results, colors
