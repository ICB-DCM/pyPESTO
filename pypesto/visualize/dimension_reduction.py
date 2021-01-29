import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from ..ensemble import Ensemble



def projection_scatter_lowlevel(projection,
                                ax: Optional[plt.Axes] = None,
                                size: Optional[Tuple[float]] = (12, 6),
                                color_by=None,
                                color_map='viridis',
                                background_color=(0., 0., 0., 1.),
                                marker_type='.',
                                scatter_size=0.2,
                                invert_scatter_order=False):
    """


    Parameters
    ----------

    projection:
        array of data points in reduced dimension

    ax:
        Axes object to use.

    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # first get the data to check identifiability
    # axes
    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)
    plt.sca(ax)

    if color_by is None:
        color_by = np.array([1.] * projection.shape[0])

    ordering = 1
    if invert_scatter_order:
        ordering = -1

    plt.scatter(projection[::ordering, 0], projection[::ordering, 1],
                c=color_by, cmap=color_map, marker=marker_type, s=scatter_size)
    ax = plt.gca()
    ax.set_facecolor(background_color)

    plt.tight_layout()

    return ax