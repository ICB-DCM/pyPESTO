from typing import List, Optional, Union

import matplotlib.cm as cm
import numpy as np

from pypesto.util import assign_clusters

# for typehints
from ..C import RGBA


def assign_clustered_colors(vals, balance_alpha=True, highlight_global=True):
    """
    Cluster and assign colors.

    Parameters
    ----------
    vals: numeric list or array
        List to be clustered and assigned colors.
    balance_alpha: bool (optional)
        Flag indicating whether alpha for large clusters should be reduced to
        avoid overplotting (default: True)
    highlight_global: bool (optional)
        flag indicating whether global optimum should be highlighted

    Returns
    -------
    colors: list of RGBA
        One for each element in 'vals'.
    """
    # sanity checks
    if vals is None or len(vals) == 0:
        return []

    # assign clusters
    clusters, cluster_size = assign_clusters(vals)

    # create list of colors, which has the correct shape
    n_clusters = 1 + max(clusters) - sum(cluster_size == 1)

    # if best value was found more than once: we need one color less
    if highlight_global and cluster_size[0] > 1:
        n_clusters -= 1

    # fill color array from colormap
    colormap = cm.ScalarMappable().to_rgba
    color_list = colormap(np.linspace(0.0, 1.0, n_clusters))

    # best optimum should be colored in red
    if highlight_global and cluster_size[0] > 1:
        color_list = np.concatenate(([[1.0, 0.0, 0.0, 1.0]], color_list))

    # We have clustered the results. However, clusters may have size 1,
    # so we need to rearrange the regroup the results into "no_clusters",
    # which will be grey, and "real_clusters", which will be colored

    # run through the cluster_indices and collect which element
    # is in 1-element-clusters, and which is in real clusters
    no_clusters = np.where(cluster_size == 1)[0]
    real_clusters = np.unique(np.where(cluster_size > 1)[0])

    # assign transparency valuesaccording to cluster size, if wanted
    if balance_alpha:
        # assign neutral color, add 1 for avoiding division by zero
        grey = [0.7, 0.7, 0.7, min(1.0, 5.0 / (no_clusters.size + 1.0))]

        # reduce alpha level depend on size of each cluster
        n_cluster_size = np.delete(cluster_size, no_clusters)
        for icluster in range(n_clusters):
            color_list[icluster][3] = min(1.0, 5.0 / n_cluster_size[icluster])
    else:
        # assign neutral color
        grey = [0.7, 0.7, 0.7, 1.0]

    # create a color list, prfilled with grey values
    colors = np.array([grey] * clusters.size)

    # assign colors to real clusters
    for icol, iclust in enumerate(real_clusters):
        # find indices belonging to the cluster iclust and assign color
        ind_of_iclust = np.argwhere(clusters == iclust).flatten()
        colors[ind_of_iclust, :] = color_list[icol, :]

    # if best value was found only once: replace it with red
    if highlight_global and cluster_size[0] == 1:
        colors[0] = [1.0, 0.0, 0.0, 1.0]

    return colors


def assign_colors(
    vals, colors=None, balance_alpha=True, highlight_global=True
):
    """
    Assign colors or format user specified colors.

    Parameters
    ----------
    vals: numeric list or array
        List to be clustered and assigned colors.

    colors: list, or RGBA, optional
        list of colors, or single color

    balance_alpha: bool (optional)
        Flag indicating whether alpha for large clusters should be reduced to
        avoid overplotting (default: True)

    highlight_global: bool (optional)
        flag indicating whether global optimum should be highlighted

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
        return assign_clustered_colors(
            vals,
            balance_alpha=balance_alpha,
            highlight_global=highlight_global,
        )

    # The user passed values and colors: parse them first!
    # we want everything to be numpy arrays, to not check every time whether a
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

    if colors.shape[1] == 4 and n_vals == colors.shape[0]:
        return colors

    if colors.shape[0] == 4:
        colors = np.transpose(colors)
        if n_vals == colors.shape[0]:
            return colors

    # Shape of array did not match n_vals. Error due to size mismatch:
    raise ValueError(
        'Incorrect color input. Colors must be specified either as '
        'list of `[r, g, b, alpha]` with length equal to that of `vals` '
        f'(here: {n_vals}), or as a single `[r, g, b, alpha]`.'
    )


def assign_colors_for_list(
    num_entries: int,
    colors: Optional[Union[RGBA, List[RGBA], np.ndarray]] = None,
) -> Union[List[List[float]], np.ndarray]:
    """
    Create a list of colors for a list of items.

    Can also check a user-provided list of colors and use this if
    everything is ok.

    Parameters
    ----------
    num_entries:
        number of results in list
    colors:
        list of colors, or single color

    Returns
    -------
    colors:
        List of RGBA, one for each element in 'vals'.
    """
    # if the user did not specify any colors:
    if colors is None:
        # default colors will be used, on for each entry in the result list.
        # Colors are created from assign_colors, which needs a dummy list
        dummy_clusters = np.array(list(range(num_entries)) * 2)

        # we don't want alpha levels for all plotting routines in this case...
        colors = assign_colors(
            dummy_clusters, balance_alpha=False, highlight_global=False
        )

        # dummy cluster had twice as many entries as really there. Reduce.
        real_indices = np.arange(int(colors.shape[0] / 2))
        return colors[real_indices]

    # if the user specified color lies does not match the number of results
    if len(colors) != num_entries:
        raise (
            'Incorrect color input. Colors must be specified either as '
            'list of [r, g, b, alpha] with length equal to function '
            'values Number of function (here: ' + str(num_entries) + '), '
            'or as one single [r, g, b, alpha] color.'
        )

    return colors
