from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ..C import COLOR
from ._style import get_ax, get_axes_array

if TYPE_CHECKING:
    try:
        import umap

        UmapTypeObject = umap.umap_.UMAP
    except ImportError:
        UmapTypeObject = None


def projection_scatter_umap(
    umap_coordinates: np.ndarray, components: Sequence[int] = (0, 1), **kwargs
):
    """
    Plot a scatter plots for UMAP coordinates.

    Creates either one or multiple scatter plots, depending on the number of
    coordinates passed to it.

    Parameters
    ----------
    umap_coordinates:
        array of umap coordinates (returned as first output by the routine
        get_umap_representation) to be shown as scatter plot

    components:
        Components to be plotted (corresponds to columns of umap_coordinates)

    Returns
    -------
    axs:
        Either one axes object, or a dictionary of plot axes (depending on the
        number of coordinates passed)
    """
    n_components = len(components)
    if n_components == 2:
        # handle components
        x_label = f"UMAP component {components[0] + 1}"
        y_label = f"UMAP component {components[1] + 1}"
        dataset = umap_coordinates[:, components]

        # call lowlevel routine
        return ensemble_scatter_lowlevel(
            dataset, x_label=x_label, y_label=y_label, **kwargs
        )
    else:
        # We got more than two components. Plot a cross-classification table
        # Create the labels first
        component_labels = [
            f"UMAP component {components[i_comp] + 1}"
            for i_comp in range(n_components)
        ]
        # reduce pca components
        dataset = umap_coordinates[:, components]
        # run lowlevel plot
        return ensemble_crosstab_scatter_lowlevel(
            dataset, component_labels, **kwargs
        )


def projection_scatter_umap_original(
    umap_object: UmapTypeObject,
    color_by: Sequence[float] = None,
    components: Sequence[int] = (0, 1),
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """
    See `projection_scatter_umap` for more documentation.

    Wrapper around umap.plot.points. Similar to `projection_scatter_umap`, but
    uses the original plotting routine from umap.plot.

    Parameters
    ----------
    umap_object:
        umap object (returned as second output by get_umap_representation)
        to be shown as scatter plot
    color_by:
        A sequence/list of floats, which specify the color in the colormap
    components:
        Components to be plotted (corresponds to columns of umap_coordinates)
    ax:
        Axes object to use.

    Returns
    -------
    ax: matplotlib.Axes
        The plot axes.
    """
    import umap.plot

    # reduce, if necessary
    umap_object.embedding_ = umap_object.embedding_[:, components]

    # use umap's original plotting routine to visualize
    if ax is not None:
        kwargs["ax"] = ax

    return umap.plot.points(
        umap_object,
        values=color_by,
        theme="viridis",
        **kwargs,
    )


def projection_scatter_pca(
    pca_coordinates: np.ndarray, components: Sequence[int] = (0, 1), **kwargs
):
    """
    Plot a scatter plot for PCA coordinates.

    Creates either one or multiple scatter plots, depending on the number of
    coordinates passed to it.

    Parameters
    ----------
    pca_coordinates:
        array of pca coordinates (returned as first output by the routine
        get_pca_representation) to be shown as scatter plot
    components:
        Components to be plotted (corresponds to columns of pca_coordinates)

    Returns
    -------
    axs:
        Either one axes object, or a dictionary of plot axes (depending on the
        number of coordinates passed)
    """
    n_components = len(components)
    if n_components == 2:
        # handle components
        x_label = f"PCA component {components[0] + 1}"
        y_label = f"PCA component {components[1] + 1}"

        dataset = pca_coordinates[:, components]

        # call lowlevel routine
        return ensemble_scatter_lowlevel(
            dataset, x_label=x_label, y_label=y_label, **kwargs
        )
    else:
        # We got more than two components. Plot a cross-classification table
        # Create the labels first
        component_labels = [
            f"PCA component {components[i_comp] + 1}"
            for i_comp in range(n_components)
        ]
        # reduce pca components
        dataset = pca_coordinates[:, components]
        # run lowlevel plot
        return ensemble_crosstab_scatter_lowlevel(
            dataset, component_labels, **kwargs
        )


def ensemble_crosstab_scatter_lowlevel(
    dataset: np.ndarray,
    component_labels: Sequence[str] = None,
    axes: np.ndarray | None = None,
    size: tuple[float, float] | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Plot cross-classification table of scatter plots for different coordinates.

    Lowlevel routine for multiple UMAP and PCA plots, but can also be used to
    visualize, e.g., parameter traces across optimizer runs.

    Parameters
    ----------
    dataset:
        array of data points to be shown as scatter plot
    component_labels:
        labels for the x-axes and the y-axes

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.
    """
    # We got more than two components. Create a cross-classification table
    n_components = dataset.shape[1]
    if component_labels is None:
        component_labels = [
            f"component {i_component + 1}"
            for i_component in range(n_components)
        ]

    if "ax" in kwargs:
        if axes is None:
            axes = kwargs.pop("ax")
        else:
            del kwargs["ax"]

    axes = _create_crosstab_axes(n_components, axes=axes, size=size)

    for x_comp in range(0, n_components - 1):
        for y_comp in range(x_comp + 1, n_components):
            # handle axis labels
            x_label = ""
            y_label = ""
            if x_comp == 0:
                y_label = component_labels[y_comp]
            if y_comp == n_components - 1:
                x_label = component_labels[x_comp]

            # extract the wanted columns
            tmp_dataset = dataset[:, [x_comp, y_comp]]

            # call lowlevel routine
            ensemble_scatter_lowlevel(
                tmp_dataset,
                x_label=x_label,
                y_label=y_label,
                ax=axes[y_comp - 1, x_comp],
                **kwargs,
            )
    return axes


def ensemble_scatter_lowlevel(
    dataset,
    ax: plt.Axes | None = None,
    size: tuple[float, float] | None = (12, 6),
    x_label: str = "component 1",
    y_label: str = "component 2",
    color_by: Sequence[float] = None,
    color_map: str = "viridis",
    background_color: COLOR = "white",
    marker_type: str = ".",
    scatter_size: float = 0.5,
    invert_scatter_order: bool = False,
):
    """
    Create a scatter plot.

    Parameters
    ----------
    dataset:
        array of data points in reduced dimension
    ax:
        Axes object to use.
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified
    x_label:
        The x-axis label
    y_label:
        The y-axis label
    color_by:
        A sequence/list of floats, which specify the color in the colormap
    color_map:
        A colormap name known to pyplot
    background_color:
        Background color of the axes object (defaults to black)
    marker_type:
        Type of plotted markers
    scatter_size:
        Size of plotted markers
    invert_scatter_order:
        Specifies the order of plotting the scatter points, can be important
        in case of overplotting

    Returns
    -------
    ax: matplotlib.Axes
        The plot axes.
    """
    ax = get_ax(ax, size)

    if color_by is None:
        color_by = np.array([1.0] * dataset.shape[0])

    ordering = 1
    if invert_scatter_order:
        ordering = -1

    ax.scatter(
        dataset[::ordering, 0],
        dataset[::ordering, 1],
        c=color_by,
        cmap=color_map,
        marker=marker_type,
        s=scatter_size,
    )

    # beautify
    ax.set_facecolor(background_color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.figure.tight_layout()

    return ax


def _create_crosstab_axes(
    n_comp: int,
    axes: np.ndarray | None = None,
    size: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Create a figure with cross-classification table of axes.

    Parameters
    ----------
    n_comp:
        number of component to be mutually compared

    Returns
    -------
    axes:
        A 2-D NumPy array of plot axes.
    """
    n_grid = n_comp - 1
    if size is None and axes is None:
        size = (3.0 * n_grid, 3.0 * n_grid)

    axes = get_axes_array(axes=axes, nrows=n_grid, ncols=n_grid, size=size)
    for ax in axes.flat:
        ax.clear()
        ax.set_visible(False)

    for x_comp in range(0, n_comp - 1):
        for y_comp in range(x_comp + 1, n_comp):
            axes[y_comp - 1, x_comp].set_visible(True)

    return axes
