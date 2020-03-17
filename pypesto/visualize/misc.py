import numpy as np
import warnings
from .clust_color import assign_colors
from .clust_color import assign_colors_for_result_list

from typing import Optional


def process_result_list(results, colors=None, legends=None):
    """
    assigns colors and legends to a list of results, chekc user provided lists

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
        colors = assign_colors_for_result_list(len(results), colors)

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
        raise ('List of results passed and list of labels do not have the'
               ' same length but should. Stopping.')

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
            offset_y = 1. - min_val
    else:
        # check whether scaling is lin or log10
        if scale_y == 'lin':
            # linear scaling doesn't need any offset
            offset_y = 0.
        else:
            # log10 scaling does need offset
            offset_y = 1. - min_val

    return offset_y


def process_y_limits(ax, y_limits):
    """
    apply user specified limits of y-axis

    Parameters
    ----------

    ax: matplotlib.Axes, optional
        Axes object to use.

    y_limits: ndarray
       y_limits, minimum and maximum, for current axes object

    min_val: float
        Smallest value to be plotted

    Returns
    -------

    ax: matplotlib.Axes, optional
        Axes object to use.
    """

    # apply y-limits, if they were specified by the user
    if y_limits is not None:
        y_limits = np.array(y_limits)
        if y_limits.size == 1:
            tmp_y_limits = ax.get_ylim()
            y_limits = [tmp_y_limits[0], y_limits]
        else:
            y_limits = [y_limits[0], y_limits[1]]

        # set limits
        ax.set_ylim(y_limits)

    return ax
