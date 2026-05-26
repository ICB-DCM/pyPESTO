from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import is_color_like

from pypesto.util import assign_clusters

# for typehints
from ..C import COLOR
from ._style import resolve_style


def _build_cluster_palette(style: dict) -> np.ndarray:
    """Sample non-best cluster colors from ``cmap_discrete``.

    Colors close to ``mle_color`` or ``outlier_color`` are filtered out so
    those reserved roles remain visually distinct from cycled cluster colors.
    """
    cmap = plt.get_cmap(style["cmap_discrete"])
    reserved = [
        np.array(mcolors.to_rgb(style["mle_color"])),
        np.array(mcolors.to_rgb(style["outlier_color"])),
    ]

    # We remove colors from the cluster palette that are too close to the
    # reserved colors (MLE red and outlier grey). The distance threshold is
    # just a reasonable heuristic.
    _RESERVED_COLOR_DISTANCE = 0.2

    # Number of evenly-spaced samples taken from a continuous cmap (e.g. viridis)
    # when ``cmap_discrete`` is set to one. Categorical cmaps (e.g. tab10) use
    # all their listed colors and ignore this.
    _CMAP_DISCRETE_SAMPLES = 10

    if hasattr(cmap, "colors"):
        candidates = [mcolors.to_rgba(c) for c in cmap.colors]
    else:
        candidates = [
            cmap(i / (_CMAP_DISCRETE_SAMPLES - 1))
            for i in range(_CMAP_DISCRETE_SAMPLES)
        ]
    palette = [
        c
        for c in candidates
        if all(
            np.linalg.norm(np.array(c[:3]) - r) > _RESERVED_COLOR_DISTANCE
            for r in reserved
        )
    ]
    if not palette:
        palette = candidates
    return np.array(palette)


def assign_clustered_colors(
    vals: np.ndarray,
    balance_alpha: bool = True,
    highlight_global: bool = True,
    style: dict | None = None,
):
    """
    Cluster and assign colors.

    Parameters
    ----------
    vals:
        List to be clustered and assigned colors.
    balance_alpha:
        Flag indicating whether alpha for large clusters should be reduced to
        avoid overplotting
    highlight_global:
        flag indicating whether global optimum should be highlighted
    style:
        Pre-resolved visualization style dict, as returned by
        :func:`pypesto.visualize._style.resolve_style`. When ``None``, defaults
        are used.

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

    if style is None:
        style = resolve_style(None)
    palette = _build_cluster_palette(style)
    mle_rgba = list(mcolors.to_rgba(style["mle_color"]))
    outlier_rgb = list(mcolors.to_rgb(style["outlier_color"]))

    # create list of colors, which has the correct shape
    n_clusters = 1 + max(clusters) - sum(cluster_size == 1)

    # if best value was found more than once: we need one color less
    if highlight_global and cluster_size[0] > 1:
        n_clusters -= 1

    # fill color array by cycling through the categorical cluster palette
    color_list = palette[np.arange(n_clusters) % len(palette)].copy()

    # best optimum should be colored in MLE red
    if highlight_global and cluster_size[0] > 1:
        color_list = np.concatenate(([mle_rgba], color_list))

    # We have clustered the results. However, clusters may have size 1,
    # so we need to rearrange the regroup the results into "no_clusters",
    # which will be grey, and "real_clusters", which will be colored

    # run through the cluster_indices and collect which element
    # is in 1-element-clusters, and which is in real clusters
    no_clusters = np.where(cluster_size == 1)[0]
    real_clusters = np.unique(np.where(cluster_size > 1)[0])

    # assign transparency values according to cluster size, if wanted
    if balance_alpha:
        # set minimal alpha value to avoid non-visible colors
        min_alpha = 0.01
        # alpha shrinks with the number of singletons to avoid overplotting
        grey = [*outlier_rgb, min(1.0, 5.0 / (no_clusters.size + 1.0))]

        # reduce alpha level depend on size of each cluster
        n_cluster_size = np.delete(cluster_size, no_clusters)
        for icluster in range(n_clusters):
            color_list[icluster][3] = min(
                1.0, max(5.0 / n_cluster_size[icluster], min_alpha)
            )
    else:
        grey = [*outlier_rgb, 1.0]

    # create a color list, prfilled with grey values
    colors = np.array([grey] * clusters.size)

    # assign colors to real clusters
    for icol, iclust in enumerate(real_clusters):
        # find indices belonging to the cluster iclust and assign color
        ind_of_iclust = np.argwhere(clusters == iclust).flatten()
        colors[ind_of_iclust, :] = color_list[icol, :]

    # if best value was found only once: replace it with MLE red
    if highlight_global and cluster_size[0] == 1:
        colors[0] = mle_rgba

    return colors


def assign_colors(
    vals: np.ndarray,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    balance_alpha: bool = True,
    highlight_global: bool = True,
    style: dict | None = None,
) -> np.ndarray:
    """
    Assign colors or format user specified colors.

    Parameters
    ----------
    vals:
        List to be clustered and assigned colors.
    colors:
        list of colors recognized by matplotlib, or single color
    balance_alpha:
        Flag indicating whether alpha for large clusters should be reduced to
        avoid overplotting
    highlight_global:
        flag indicating whether global optimum should be highlighted
    style:
        Pre-resolved visualization style dict, as returned by
        :func:`pypesto.visualize._style.resolve_style`. When ``None``, defaults
        are used.

    Returns
    -------
    colors: list of colors recognized by matplotlib
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
            style=style,
        )

    # Get number of elements and use user assigned colors
    n_vals = len(vals) if isinstance(vals, list) else vals.size

    if is_color_like(colors):
        # one color was passed
        return np.array([colors] * n_vals)

    elif (
        isinstance(colors, (list, np.ndarray))
        and len(colors) == len(vals)
        and sum([is_color_like(c) for c in colors]) == len(colors)
    ):
        # a list of colors was passed, one for each value in vals

        # convert to ndarray
        colors = np.array(colors)
        return colors

    # Shape of array did not match n_vals. Error due to size mismatch:
    raise ValueError(
        "Incorrect color input. Colors must be specified either as "
        "list of colors recognized by matplotlib with length equal to that of `vals` "
        f"(here: {n_vals}), or as a single matplotlib color."
    )


def assign_colors_for_list(
    num_entries: int,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    style: dict | None = None,
) -> list[list[float]] | np.ndarray:
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
    style:
        Pre-resolved visualization style dict, as returned by
        :func:`pypesto.visualize._style.resolve_style`. When ``None``, defaults
        are used.

    Returns
    -------
    colors:
        List of RGBA, one for each element in 'vals'.
    """
    # if the user did not specify any colors:
    if colors is None:
        # default colors will be used, one for each entry in the result list.
        # Colors are created from assign_colors, which needs a dummy list
        # doubled for clustering
        dummy_clusters = np.array(list(range(num_entries)) * 2)

        # we don't want alpha levels for all plotting routines in this case...
        colors = assign_colors(
            dummy_clusters,
            balance_alpha=False,
            highlight_global=False,
            style=style,
        )

        # dummy cluster had twice as many entries as really there. Reduce.
        real_indices = np.arange(0, colors.shape[0], 2)
        return colors[real_indices]

    # Pass the colors through assign_colors to check correct format of RGBA
    return assign_colors(
        vals=np.array(list(range(num_entries))),
        colors=colors,
        balance_alpha=False,
        highlight_global=False,
        style=style,
    )
