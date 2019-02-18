import numpy as np
import warnings
from .clust_color import assign_colors
from .clust_color import assign_colors_for_result_list


def handle_result_list(results, colors=None, legends=None):
    """
    assigns colors to a list of results

    Parameters
    ----------

    results: list or pypesto.Result
        list of pypesto.Result objects or a single pypesto.Result

    colors: list, optional
        list of RGB colors

    legends: str or list
        labels for line plots

    Returns
    -------

    results: pypesto.result or list
       list of pypesto.result objects

    colors: list of RGB
        One for each element in 'results'.

    legends: str or list
        labels for line plots
    """

    # check if list
    if not isinstance(results, list):
        # assign colors
        colors = assign_colors([results], colors)

        # create lists from results and colors for correct later handling
        results = [results]
        colors = [colors]
        if not isinstance(legends, list):
            legends = [legends]
    else:
        # if more than one result is passed, one color per result is used
        if colors is None:
            colors = assign_colors_for_result_list(len(results), colors)

        # if the user passed a list of colors, does it have the correct length?
        if any(isinstance(i_color, list) for i_color in colors):
            if len(colors) != len(colors):
                raise ('List of results and list of colors is passed. The '
                       'length of the color list must match he length of the '
                       'results list. Stopping.')

        # check whether list of legends has the correct length
        if legends is None:
            # create some custom legends
            legends = []
            for i_leg in range(len(results)):
                legends.append(['Result ' + str(i_leg)])

        # errors due to incorrect lengths
        if not isinstance(legends, list) and len(results) > 1:
            raise ('List of results passed, but only one label. Please pass a '
                   'list of labels with the same length as the list of '
                   'results. Stopping.')
        if len(legends) != len(results):
            raise ('List of results passed and list of labels do not have the'
                   ' same length but should. Stopping.')

    return results, colors, legends


def handle_offset_y(offset_y, scale_y, min_val):

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


def handle_y_limits(ax, y_limits):
    # handle y-limits
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
