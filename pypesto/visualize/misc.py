from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np

from ..C import (
    ALL,
    ALL_CLUSTERED,
    COLOR,
    FIRST_CLUSTER,
    FREE_ONLY,
    LEN_RGB,
    LEN_RGBA,
    RGB,
    RGB_RGBA,
    RGBA_ALPHA,
    RGBA_MAX,
    RGBA_MIN,
    RGBA_WHITE,
)
from ..result import Result
from ..util import assign_clusters, delete_nan_inf
from .clust_color import assign_colors_for_list

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

logger = logging.getLogger(__name__)


def process_result_list(
    results: Result | list[Result],
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: str | list[str] | None = None,
) -> tuple[list[Result], list[COLOR], list[str]]:
    """
    Assign colors and legends to a list of results, check user provided lists.

    Parameters
    ----------
    results:
        list of pypesto.Result objects or a single pypesto.Result
    colors:
        list of colors recognized by matplotlib, or single color
    legends:
        labels for line plots

    Returns
    -------
    results:
       list of pypesto.Result objects
    colors:
        One for each element in 'results'.
    legends:
        labels for line plots
    """
    # check how many results were passed
    single_result = False
    legend_type_error = False
    if isinstance(results, list):
        if len(results) == 1:
            single_result = True
    else:
        single_result = True
        results = [results]

    # handle results according to their number
    if single_result:
        # assign colors and create list for later handling
        if colors is not None and isinstance(colors, list):
            colors = [np.array(colors)]
        else:
            colors = [colors]

        # create list of legends for later handling
        if not isinstance(legends, list):
            legends = [legends]
        try:
            str(legends[0])
        except TypeError:
            legend_type_error = True
    else:
        # if more than one result is passed, we use one color per result
        colors = assign_colors_for_list(len(results), colors)

        # check whether list of legends has the correct length
        if legends is None:
            # No legends were passed: create some custom legends
            legends = []
            for i_leg in range(len(results)):
                legends.append("Result " + str(i_leg))
        else:
            # legends were passed by user: check length
            try:
                if isinstance(legends, str):
                    legends = [legends]
                if len(legends) != len(results):
                    raise ValueError(
                        "List of results passed and list of labels do "
                        "not have the same length."
                    )
            except TypeError:
                legend_type_error = True

    if legend_type_error:
        raise TypeError("Unexpected legend type.")

    return results, colors, legends


def process_offset_y(
    offset_y: float | None, scale_y: str, min_val: float
) -> float:
    """
    Compute offset for y-axis, depend on user settings.

    Parameters
    ----------
    offset_y:
       value for offsetting the later plotted values, in order to ensure
       positivity if a semilog-plot is used
    scale_y:
       Can be 'lin' or 'log10', specifying whether values should be plotted
       on linear or on log10-scale
    min_val:
        Smallest value to be plotted

    Returns
    -------
    offset_y: float
       value for offsetting the later plotted values, in order to ensure
       positivity if a semilog-plot is used
    """
    # check whether the offset specified by the user is sufficient
    if offset_y is not None:
        if (scale_y == "log10") and (min_val + offset_y <= 0.0):
            warnings.warn(
                "Offset specified by user is insufficient. "
                "Ignoring specified offset and using "
                + str(np.abs(min_val) + 1.0)
                + " instead.",
                stacklevel=2,
            )
        else:
            return offset_y
    else:
        # check whether scaling is lin or log10
        if scale_y == "lin":
            # linear scaling doesn't need any offset
            return 0.0

    return 1.0 - min_val


def process_y_limits(
    ax: Axes,
    y_limits: None | Iterable[float] | np.ndarray,
) -> Axes:
    """
    Apply user specified limits of y-axis.

    Parameters
    ----------
    ax:
        Axes object to use.
    y_limits:
       y_limits, minimum and maximum, for current axes object

    Returns
    -------
    ax:
        Axes object to use.
    """
    # apply y-limits, if they were specified by the user
    if y_limits is not None:
        y_limits = np.array(y_limits)

        # check validity of bounds
        if y_limits.size == 0:
            y_limits = np.array(ax.get_ylim())
        elif y_limits.size == 1:
            # if the user specified only an upper bound
            tmp_y_limits = ax.get_ylim()
            y_limits = [tmp_y_limits[0], y_limits]
        elif y_limits.size > 1:
            y_limits = [y_limits[0], y_limits[1]]

        # check validity of bounds if plotting in log-scale
        if ax.get_yscale() == "log" and y_limits[0] <= 0.0:
            tmp_y_limits = ax.get_ylim()
            if y_limits[1] <= 0.0:
                y_limits = tmp_y_limits
                warnings.warn(
                    "Invalid bounds for plotting in "
                    "log-scale. Using defaults bounds.",
                    stacklevel=2,
                )
            else:
                y_limits = [tmp_y_limits[0], y_limits[1]]
                warnings.warn(
                    "Invalid lower bound for plotting in "
                    "log-scale. Using only upper bound.",
                    stacklevel=2,
                )

        # set limits
        ax.set_ylim(y_limits)

    else:
        # No limits passed, but if we have a result list: check the limits
        ax_limits = np.array(ax.get_ylim())
        data_limits = ax.dataLim.ymin, ax.dataLim.ymax

        # Check if data fits to axes and adapt limits, if necessary
        if np.isfinite(data_limits).all() and (
            ax_limits[0] > data_limits[0] or ax_limits[1] < data_limits[1]
        ):
            # Get range of data
            data_range = data_limits[1] - data_limits[0]
            if ax.get_yscale() == "log":
                data_range = np.log10(data_range)
                new_limits = (
                    np.power(10, np.log10(data_limits[0]) - 0.02 * data_range),
                    np.power(10, np.log10(data_limits[1]) + 0.02 * data_range),
                )
            else:
                new_limits = (
                    data_limits[0] - 0.02 * data_range,
                    data_limits[1] + 0.02 * data_range,
                )

            # set limits
            ax.set_ylim(new_limits)

    return ax


def rgba2rgb(fg: RGB_RGBA, bg: RGB_RGBA = None) -> RGB:
    """Combine two colors, removing transparency.

    Parameters
    ----------
    fg:
        Foreground color.
    bg:
        Background color.

    Returns
    -------
    The combined color.
    """
    if bg is None:
        bg = RGBA_WHITE
    if len(bg) == LEN_RGBA:
        # return foreground if background is fully transparent
        if bg[RGBA_ALPHA] == RGBA_MIN:
            return fg
    else:
        if len(bg) != LEN_RGB:
            raise IndexError(
                "A background color of unexpected length was provided: {bg}"
            )
        bg = (*bg, RGBA_MAX)

    # return the foreground color if has no transparency
    if len(fg) == LEN_RGB or fg[RGBA_ALPHA] == RGBA_MAX:
        return fg
    if len(fg) != LEN_RGBA:
        raise IndexError(
            "A foreground color of unexpected length was provided: {fg}"
        )

    def apparent_composite_color_component(
        fg_component: float,
        bg_component: float,
        fg_alpha: float = fg[RGBA_ALPHA],
        bg_alpha: float = bg[RGBA_ALPHA],
    ) -> float:
        """
        Composite a foreground over a background color component.

        Porter and Duff equations are used for alpha compositing.

        Parameters
        ----------
        fg_component:
            The foreground color component.
        bg_component:
            The background color component.
        fg_alpha:
            The foreground color transparency/alpha component.
        bg_alpha:
            The background color transparency/alpha component.

        Returns
        -------
        The component of the new color.
        """
        return (
            fg_component * fg_alpha
            + bg_component * bg_alpha * (RGBA_MAX - fg_alpha)
        ) / (fg_alpha + bg_alpha * (RGBA_MAX - fg_alpha))

    return [
        apparent_composite_color_component(fg[i], bg[i])
        for i in range(LEN_RGB)
    ]


def process_start_indices(
    result: Result,
    start_indices: str | int | Iterable[int] = None,
) -> np.ndarray:
    """
    Process the start_indices.

    Create an array of indices if a number was provided, checks that the indices
    do not exceed the max_index and removes starts with non-finite fval.

    Parameters
    ----------
    start_indices:
        list of indices or int specifying an endpoint of the sequence of
        indices. Furthermore the following strings are possible:
            * 'all', this is the default, using all start indices.
            * 'all_clustered', this includes the best start and all clusters
            with the size > 1.
            * 'first_cluster', includes all starts that belong to the first
            cluster.
    result:
        Result to determine maximum allowed length and/or clusters.
    """
    if start_indices is None:
        start_indices = ALL
    if isinstance(start_indices, str):
        if start_indices == ALL:
            start_indices = np.asarray(range(len(result.optimize_result)))
        elif start_indices == ALL_CLUSTERED:
            clust_ind, clust_size = assign_clusters(
                delete_nan_inf(result.optimize_result.fval)[1]
            )
            # get all clusters that have size >= 2 and cluster of best start:
            clust_gr2 = np.where(clust_size > 2)[0]
            clust_gr2 = (
                np.append(clust_gr2, 0) if 0 not in clust_gr2 else clust_gr2
            )
            start_indices = np.concatenate(
                [np.where(clust_ind == i_clust)[0] for i_clust in clust_gr2]
            )
            start_indices = start_indices
        elif start_indices == FIRST_CLUSTER:
            clust_ind = assign_clusters(
                delete_nan_inf(result.optimize_result.fval)[1]
            )[0]
            start_indices = np.where(clust_ind == 0)[0]
        else:
            raise ValueError(
                f"Permissible values for start_indices are {ALL}, "
                f"{ALL_CLUSTERED}, {FIRST_CLUSTER}, an integer or a "
                f"list of indices. Got {start_indices}."
            )
    # if it is an integer n, select the first n starts
    if isinstance(start_indices, Number):
        start_indices = range(int(start_indices))

    # filter out the indices that exceed the range of possible start indices
    start_indices = [
        start_index
        for start_index in start_indices
        if start_index < len(result.optimize_result)
    ]

    # filter out the indices that are not finite
    start_indices_unfiltered = len(start_indices)
    start_indices = [
        start_index
        for start_index in start_indices
        if np.isfinite(result.optimize_result[start_index].fval)
    ]
    if len(start_indices) != start_indices_unfiltered:
        logger.warning(
            "Some start indices were removed due to inf or nan function values."
        )

    return np.asarray(start_indices, dtype=int)


def process_parameter_indices(
    result: Result,
    parameter_indices: str | Iterable[int] = FREE_ONLY,
) -> list:
    """
    Process the parameter indices, always returning a valid array.

    Create an array of indices depending on the string that is provided. Or
    returns the sequence in case a sequence was provided.

    Parameters
    ----------
    result:
        Result to determine maximum allowed length and/or clusters.
    parameter_indices:
        list of indices or str specifying the desired indices. Default is
        `free_only`. Other option is 'all', which included all estimated
        and fixed parameters.
    """
    if isinstance(parameter_indices, str):
        if parameter_indices == ALL:
            return list(range(0, result.problem.dim_full))
        elif parameter_indices == FREE_ONLY:
            return result.problem.x_free_indices
        else:
            raise ValueError(
                "Permissible values for parameter_indices are "
                f"{ALL}, {FREE_ONLY} or a list of indices."
            )
    return list(parameter_indices)
