from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.axes
import numpy as np

from .misc import get_ax, get_axes_array, hide_unused_axes
from ._style import (
    COLORBAR_WIDTH,
    GRID_SIZE_PER_COL,
    GRID_SIZE_PER_ROW,
    add_colorbar,
    resolve_style,
)

if TYPE_CHECKING:
    try:
        import umap

        UmapTypeObject = umap.umap_.UMAP
    except ImportError:
        UmapTypeObject = None


def projection_scatter_umap(
    umap_coordinates: np.ndarray,
    components: Sequence[int] = (0, 1),
    ax: matplotlib.axes.Axes | None = None,
    axes: np.ndarray | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = "UMAP projection",
    color_by: Sequence[float] | None = None,
    color_label: str = "",
    marker_type: str = "o",
    invert_scatter_order: bool = False,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes | np.ndarray:
    """
    Plot a scatter plot for UMAP coordinates.

    Creates either one scatter plot (2 components) or a cross-classification
    grid of scatter plots (more than 2 components).

    Parameters
    ----------
    umap_coordinates:
        Array of UMAP coordinates, as returned by
        :func:`~pypesto.ensemble.get_umap_representation`.  Shape
        ``(n_samples, n_umap_components)``.
    components:
        Which columns of *umap_coordinates* to plot.  With 2 entries a single
        scatter is drawn; with 3+ entries a pairwise grid is drawn.
    ax:
        Axes to draw on.  Only used when ``len(components) == 2``.  When
        ``None`` a new figure is created.
    axes:
        2-D NumPy array of Axes.  Only used when ``len(components) > 2``.
        When ``None`` a new figure with a pairwise grid is created.
    size:
        Figure size ``(width, height)`` in inches.  Ignored when *ax* / *axes*
        is supplied.  Defaults to matplotlib's default figure size for 2
        components, or a grid-scaled size for more components.
    title:
        Plot title. Applied as an axes title for 2 components and as a figure
        title for 3+ components. Pass ``None`` to suppress.
    color_by:
        A sequence of floats (length = number of samples) used to colour each
        point via the colormap.  Typical use: pass objective values so the
        lowest-fval sample is highlighted.  When ``None`` all points share the
        same neutral colour.
    color_label:
        Colorbar label shown when *color_by* is provided.
    marker_type:
        Marker style string (default ``"o"``).
    invert_scatter_order:
        When ``True`` points are plotted in reversed row order — useful when
        *color_by* encodes a ranking and you want the best-ranked points drawn
        on top.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``scatter_size``, ``scatter_alpha``, ``scatter_linewidths``,
          ``scatter_edgecolors``, ``scatter_zorder`` — scatter point geometry.
        - ``cmap_fval`` — colormap applied to ``color_by`` values.
        - ``scatter_color`` — flat point color used when ``color_by`` is
          ``None``.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    axs:
        A single :class:`matplotlib.axes.Axes` (2 components) or a 2-D NumPy
        array of Axes (more than 2 components).
    """
    n_components = len(components)
    if n_components == 2:
        dataset = umap_coordinates[:, components]
        ax = ensemble_scatter_lowlevel(
            dataset,
            ax=ax,
            size=size,
            x_label=f"UMAP component {components[0] + 1}",
            y_label=f"UMAP component {components[1] + 1}",
            color_by=color_by,
            color_label=color_label,
            marker_type=marker_type,
            invert_scatter_order=invert_scatter_order,
            title=None,
            style_kwargs=style_kwargs,
        )
        if title is not None:
            ax.set_title(title)
        return ax
    else:
        component_labels = [
            f"UMAP component {components[i_comp] + 1}"
            for i_comp in range(n_components)
        ]
        dataset = umap_coordinates[:, components]
        return ensemble_crosstab_scatter_lowlevel(
            dataset,
            component_labels,
            axes=axes,
            size=size,
            color_by=color_by,
            color_label=color_label,
            marker_type=marker_type,
            invert_scatter_order=invert_scatter_order,
            title=title,
            style_kwargs=style_kwargs,
        )


def projection_scatter_umap_original(
    umap_object: UmapTypeObject,
    color_by: Sequence[float] | None = None,
    components: Sequence[int] = (0, 1),
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = "UMAP projection",
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot UMAP coordinates using umap.plot's own rendering engine.

    Unlike :func:`projection_scatter_umap`, this wrapper delegates entirely to
    :func:`umap.plot.points` and therefore does not apply pyPESTO's
    ``style_kwargs`` system.  Use :func:`projection_scatter_umap` for a
    fully-harmonised plot; use this function only when you need umap.plot's
    specific visual style (e.g. its density shading or hover interactivity).

    Parameters
    ----------
    umap_object:
        Fitted UMAP object, as returned as the second output by
        :func:`~pypesto.ensemble.get_umap_representation`.
    color_by:
        A sequence/list of floats used to colour points via umap.plot's
        ``values`` argument.
    components:
        Which embedding columns to plot.
    ax:
        Axes to draw on.  Passed through to umap.plot as ``ax``.
    title:
        Axes title. Pass ``None`` to suppress.

    Returns
    -------
    ax: matplotlib.axes.Axes
        The plot axes.
    """
    import umap.plot

    original_embedding = umap_object.embedding_
    umap_object.embedding_ = original_embedding[:, components]
    if ax is not None:
        kwargs["ax"] = ax
    try:
        ax = umap.plot.points(
            umap_object, values=color_by, theme="viridis", **kwargs
        )
    finally:
        umap_object.embedding_ = original_embedding
    if title is not None:
        ax.set_title(title)
    return ax


def projection_scatter_pca(
    pca_coordinates: np.ndarray,
    components: Sequence[int] = (0, 1),
    ax: matplotlib.axes.Axes | None = None,
    axes: np.ndarray | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = "PCA projection",
    color_by: Sequence[float] | None = None,
    color_label: str = "",
    marker_type: str = "o",
    invert_scatter_order: bool = False,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes | np.ndarray:
    """
    Plot a scatter plot for PCA coordinates.

    Creates either one scatter plot (2 components) or a cross-classification
    grid of scatter plots (more than 2 components).

    Parameters
    ----------
    pca_coordinates:
        Array of PCA coordinates, as returned by
        :func:`~pypesto.ensemble.get_pca_representation`.  Shape
        ``(n_samples, n_pca_components)``.
    components:
        Which columns of *pca_coordinates* to plot.  With 2 entries a single
        scatter is drawn; with 3+ entries a pairwise grid is drawn, showing
        every pair of the selected components.
    ax:
        Axes to draw on.  Only used when ``len(components) == 2``.  When
        ``None`` a new figure is created.
    axes:
        2-D NumPy array of Axes.  Only used when ``len(components) > 2``.
        When ``None`` a new figure with a pairwise grid is created.
    size:
        Figure size ``(width, height)`` in inches.  Ignored when *ax* / *axes*
        is supplied.  Defaults to matplotlib's default figure size for 2
        components, or a grid-scaled size for more components.
    title:
        Plot title. Applied as an axes title for 2 components and as a figure
        title for 3+ components. Pass ``None`` to suppress.
    color_by:
        A sequence of floats (length = number of samples) used to colour each
        point via the colormap.  Typical use: pass objective values so the
        lowest-fval sample is highlighted.  When ``None`` all points share the
        same neutral colour.
    color_label:
        Colorbar label shown when *color_by* is provided.
    marker_type:
        Marker style string (default ``"o"``).
    invert_scatter_order:
        When ``True`` points are plotted in reversed row order — useful when
        *color_by* encodes a ranking and you want the best-ranked points drawn
        on top.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``scatter_size``, ``scatter_alpha``, ``scatter_linewidths``,
          ``scatter_edgecolors``, ``scatter_zorder`` — scatter point geometry.
        - ``cmap_fval`` — colormap applied to ``color_by`` values.
        - ``scatter_color`` — flat point color used when ``color_by`` is
          ``None``.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    axs:
        A single :class:`matplotlib.axes.Axes` (2 components) or a 2-D NumPy
        array of Axes (more than 2 components).
    """
    n_components = len(components)
    if n_components == 2:
        dataset = pca_coordinates[:, components]
        ax = ensemble_scatter_lowlevel(
            dataset,
            ax=ax,
            size=size,
            x_label=f"PCA component {components[0] + 1}",
            y_label=f"PCA component {components[1] + 1}",
            color_by=color_by,
            color_label=color_label,
            marker_type=marker_type,
            invert_scatter_order=invert_scatter_order,
            title=None,
            style_kwargs=style_kwargs,
        )
        if title is not None:
            ax.set_title(title)
        return ax
    else:
        component_labels = [
            f"PCA component {components[i_comp] + 1}"
            for i_comp in range(n_components)
        ]
        dataset = pca_coordinates[:, components]
        return ensemble_crosstab_scatter_lowlevel(
            dataset,
            component_labels,
            axes=axes,
            size=size,
            color_by=color_by,
            color_label=color_label,
            marker_type=marker_type,
            invert_scatter_order=invert_scatter_order,
            title=title,
            style_kwargs=style_kwargs,
        )


def ensemble_crosstab_scatter_lowlevel(
    dataset: np.ndarray,
    component_labels: Sequence[str] | None = None,
    axes: np.ndarray | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = None,
    color_by: Sequence[float] | None = None,
    color_label: str = "",
    marker_type: str = "o",
    invert_scatter_order: bool = False,
    style_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Plot cross-classification table of scatter plots for different coordinates.

    Lowlevel routine for multi-component UMAP and PCA plots, but can also be
    used to visualise parameter traces across optimizer runs.

    Parameters
    ----------
    dataset:
        Array of data points, shape ``(n_samples, n_components)``.
    component_labels:
        Labels for the x-axes and the y-axes, one per column of *dataset*.
    axes:
        Pre-existing 2-D NumPy array of Axes.  When ``None`` a new figure is
        created.
    size:
        Figure size ``(width, height)`` in inches.  Ignored when *axes* is
        supplied.  Auto-scaled to ``3 * (n_components - 1)`` per side when
        ``None``.
    title:
        Figure title.
    color_by:
        A sequence of floats (length = ``n_samples``) used to colour each
        point via ``style["cmap_fval"]``.  Applied identically in every panel.
        When provided a shared colorbar is added to the right of the grid.
    color_label:
        Shared colorbar label shown when *color_by* is provided.
    marker_type:
        Marker style string (default ``"o"``).
    invert_scatter_order:
        When ``True`` points are plotted in reversed row order.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``scatter_size``, ``scatter_alpha``, ``scatter_linewidths``,
          ``scatter_edgecolors``, ``scatter_zorder`` — scatter point geometry.
        - ``cmap_fval`` — colormap applied to ``color_by`` values.
        - ``scatter_color`` — flat point color used when ``color_by`` is
          ``None``.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.
    """
    n_components = dataset.shape[1]
    if component_labels is None:
        component_labels = [
            f"component {i + 1}" for i in range(n_components)
        ]

    if size is None and axes is None and color_by is not None:
        # extra width for the shared colorbar; without one, the grid default
        # from get_axes_array is used as-is
        n_grid = n_components - 1
        size = (
            GRID_SIZE_PER_COL * n_grid + COLORBAR_WIDTH,
            GRID_SIZE_PER_ROW * n_grid,
        )

    axes = _create_crosstab_axes(n_components, axes=axes, size=size)

    for x_comp in range(0, n_components - 1):
        for y_comp in range(x_comp + 1, n_components):
            x_label = component_labels[x_comp] if y_comp == n_components - 1 else ""
            y_label = component_labels[y_comp] if x_comp == 0 else ""
            ensemble_scatter_lowlevel(
                dataset[:, [x_comp, y_comp]],
                x_label=x_label,
                y_label=y_label,
                ax=axes[y_comp - 1, x_comp],
                color_by=color_by,
                marker_type=marker_type,
                invert_scatter_order=invert_scatter_order,
                style_kwargs=style_kwargs,
            )

    if color_by is not None:
        style = resolve_style(style_kwargs)
        add_colorbar(
            axes.flat[0].figure,
            axes,
            np.asarray(color_by),
            label=color_label,
            cmap=style["cmap_fval"],
        )

    if title is not None:
        axes.flat[0].figure.suptitle(title)

    return axes


def ensemble_scatter_lowlevel(
    dataset: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = None,
    x_label: str = "component 1",
    y_label: str = "component 2",
    color_by: Sequence[float] | None = None,
    color_label: str = "",
    marker_type: str = "o",
    invert_scatter_order: bool = False,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
    """
    Create a single scatter plot panel.

    Parameters
    ----------
    dataset:
        Array of shape ``(n_samples, 2)``.
    ax:
        Axes to draw on.  When ``None`` a new figure is created.
    size:
        Figure size ``(width, height)`` in inches.  Only used when *ax* is
        ``None``.  Defaults to matplotlib's default figure size (plus
        ``COLORBAR_WIDTH`` when a colorbar is drawn).
    title:
        Axes title. Pass ``None`` to suppress.
    x_label:
        X-axis label.
    y_label:
        Y-axis label.
    color_by:
        A sequence of floats (one per sample) encoding each point's colour via
        ``style["cmap_fval"]``.  When provided and *ax* was ``None`` a
        colorbar is added automatically and the figure is widened by
        ``COLORBAR_WIDTH``.  When ``None`` all points share the same neutral
        colour.
    color_label:
        Colorbar label shown when *color_by* is provided.
    marker_type:
        Marker style string (default ``"o"``).
    invert_scatter_order:
        When ``True`` points are plotted in reversed row order — useful when
        *color_by* encodes a ranking and you want the best-ranked points drawn
        on top.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``scatter_size``, ``scatter_alpha``, ``scatter_linewidths``,
          ``scatter_edgecolors``, ``scatter_zorder`` — scatter point geometry.
        - ``cmap_fval`` — colormap applied to ``color_by`` values.
        - ``scatter_color`` — flat point color used when ``color_by`` is
          ``None``.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    ax: matplotlib.axes.Axes
        The plot axes.
    """
    style = resolve_style(style_kwargs)

    ax_was_none = ax is None
    add_cb = color_by is not None and ax_was_none
    if add_cb:
        # need a concrete width to add the colorbar allowance to
        base = size if size is not None else matplotlib.rcParams["figure.figsize"]
        size = (base[0] + COLORBAR_WIDTH, base[1])

    ax = get_ax(ax, size)

    ordering = -1 if invert_scatter_order else 1

    scatter_kwargs: dict = dict(
        marker=marker_type,
        s=style["scatter_size"],
        alpha=style["scatter_alpha"],
        linewidths=style["scatter_linewidths"],
        edgecolors=style["scatter_edgecolors"],
        zorder=style["scatter_zorder"],
    )
    if color_by is not None:
        scatter_kwargs["c"] = np.asarray(color_by)[::ordering]
        scatter_kwargs["cmap"] = style["cmap_fval"]
    else:
        scatter_kwargs["color"] = style["scatter_color"]

    sc = ax.scatter(dataset[::ordering, 0], dataset[::ordering, 1], **scatter_kwargs)

    if add_cb:
        add_colorbar(
            ax.figure,
            np.array([[ax]]),
            np.asarray(color_by),
            label=color_label,
            cmap=style["cmap_fval"],
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)

    return ax


def _create_crosstab_axes(
    n_comp: int,
    axes: np.ndarray | None = None,
    size: tuple[float, float] | None = None,
) -> np.ndarray:
    """Create a figure with a cross-classification grid of axes."""
    n_grid = n_comp - 1
    axes = get_axes_array(axes=axes, nrows=n_grid, ncols=n_grid, size=size)
    used_indices = [
        (y_comp - 1) * n_grid + x_comp
        for x_comp in range(0, n_comp - 1)
        for y_comp in range(x_comp + 1, n_comp)
    ]
    return hide_unused_axes(axes=axes, used_indices=used_indices, clear=True)
