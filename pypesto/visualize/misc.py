import numpy as np
import warnings
from .clust_color import assign_colors
from .clust_color import assign_colors_for_list

from numbers import Number
from typing import Iterable, List, Optional, Union

from .constants import (
    LEN_RGB,
    LEN_RGBA,
    RGB,
    RGB_RGBA,
    RGBA_MIN,
    RGBA_MAX,
    RGBA_ALPHA,
    RGBA_WHITE,
)


def process_result_list(results, colors=None, legends=None):
    """
    assigns colors and legends to a list of results, check user provided lists

    Parameters
    ----------

    results: list or pypesto.Result
        list of pypesto.Result objects or a single pypesto.Result

    colors: list, optional
        list of RGBA colors

    legends: str or list
        labels for line plots

    Returns
    -------

    results: list of pypesto.Result
       list of pypesto.Result objects

    colors: list of RGBA
        One for each element in 'results'.

    legends: list of str
        labels for line plots
    """

    # check how many results were passed
    single_result = False
    legend_error = False
    if isinstance(results, list):
        if len(results) == 1:
            single_result = True
    else:
        single_result = True
        results = [results]

    # handle results according to their number
    if single_result:
        # assign colors and create list for later handling
        if colors is not None:
            colors = assign_colors(results, colors)
        colors = [colors]

        # create list of legends for later handling
        if not isinstance(legends, list):
            legends = [legends]
    else:
        # if more than one result is passed, we use one color per result
        colors = assign_colors_for_list(len(results), colors)

        # check whether list of legends has the correct length
        if legends is None:
            # No legends were passed: create some custom legends
            legends = []
            for i_leg in range(len(results)):
                legends.append('Result ' + str(i_leg))
        else:
            # legends were passed by user: check length
            if isinstance(legends, list):
                if len(legends) != len(results):
                    legend_error = True
            else:
                legend_error = True

    # size of legend list and size of results does not match
    if legend_error:
        raise ValueError('List of results passed and list of labels do '
                         'not have the same length but should. Stopping.')

    return results, colors, legends


def process_offset_y(offset_y: Optional[float],
                     scale_y: str,
                     min_val: float) -> float:
    """
    compute offset for y-axis, depend on user settings

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
        if (scale_y == 'log10') and (min_val + offset_y <= 0.):
            warnings.warn("Offset specified by user is insufficient. "
                          "Ignoring specified offset and using " +
                          str(np.abs(min_val) + 1.) + " instead.")
        else:
            return offset_y
    else:
        # check whether scaling is lin or log10
        if scale_y == 'lin':
            # linear scaling doesn't need any offset
            return 0.

    return 1. - min_val


def process_y_limits(ax, y_limits):
    """
    apply user specified limits of y-axis

    Parameters
    ----------

    ax: matplotlib.Axes, optional
        Axes object to use.

    y_limits: ndarray
       y_limits, minimum and maximum, for current axes object

    Returns
    -------

    ax: matplotlib.Axes, optional
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
        if ax.get_yscale() == 'log' and y_limits[0] <= 0.:
            tmp_y_limits = ax.get_ylim()
            if y_limits[1] <= 0.:
                y_limits = tmp_y_limits
                warnings.warn("Invalid bounds for plotting in "
                              "log-scale. Using defaults bounds.")
            else:
                y_limits = [tmp_y_limits[0], y_limits[1]]
                warnings.warn("Invalid lower bound for plotting in "
                              "log-scale. Using only upper bound.")

            # set limits
            ax.set_ylim(y_limits)

    else:
        # No limits passed, but if we have a result list: check the limits
        ax_limits = np.array(ax.get_ylim())
        data_limits = ax.dataLim.ymin, ax.dataLim.ymax

        # Check if data fits to axes and adapt limits, if necessary
        if ax_limits[0] > data_limits[0] or ax_limits[1] < data_limits[1]:
            # Get range of data
            data_range = data_limits[1] - data_limits[0]
            if ax.get_yscale() == 'log':
                data_range = np.log10(data_range)
                new_limits = (
                    np.power(10, np.log10(data_limits[0]) - 0.02 * data_range),
                    np.power(10, np.log10(data_limits[1]) + 0.02 * data_range))
            else:
                new_limits = (data_limits[0] - 0.02 * data_range,
                              data_limits[1] + 0.02 * data_range)

            # set limits
            ax.set_ylim(new_limits)

    return ax


def process_start_indices(start_indices: Union[int, Iterable[int]],
                          max_length: int) -> List[int]:
    """
    helper function that processes the start_indices and
    creates an array of indices if a number was provided and checks that the
    indices do not exceed the max_index

    Parameters
    ----------
    start_indices:
        list of indices or int specifying an endpoint of the sequence of
        indices
    max_length:
        maximum possible index for the start_indices
    """

    if isinstance(start_indices, Number):
        start_indices = range(int(start_indices))

    start_indices = np.array(start_indices, dtype=int)

    # check, whether index set is not too big
    start_indices = [start_index for start_index in start_indices if
                     start_index < max_length]

    return start_indices


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
                'A background color of unexpected length was provided: {bg}'
            )
        bg = (*bg, RGBA_MAX)

    # return the foreground color if has no transparency
    if len(fg) == LEN_RGB or fg[RGBA_ALPHA] == RGBA_MAX:
        return fg
    if len(fg) != LEN_RGBA:
        raise IndexError(
            'A foreground color of unexpected length was provided: {fg}'
        )

    def apparent_composite_color_component(
            fg_component: float,
            bg_component: float,
            fg_alpha: float = fg[RGBA_ALPHA],
            bg_alpha: float = bg[RGBA_ALPHA],
    ) -> float:
        """
        Composite a foreground color component over a background color
        component.

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
            fg_component * fg_alpha +
            bg_component * bg_alpha * (RGBA_MAX - fg_alpha)
        ) / (fg_alpha + bg_alpha * (RGBA_MAX - fg_alpha))

    return [
        apparent_composite_color_component(fg[i], bg[i])
        for i in range(LEN_RGB)
    ]
