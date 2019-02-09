import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .visualization import ReferencePoint
from .visualization import VisualizationOptions
from .clust_color import assign_clustered_colors
# from .visualization import apply_options

def waterfall(result, ax=None,
              size=(18.5, 10.5),
              options=None,
              reference=None):
    """
    Plot waterfall plot.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    options: VisualizationOptions, optional
        Options specifying axes, colors and reference points

    reference: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # extract cost function values from result
    fvals = result.optimize_result.get_for_key('fval')

    # call lowlevel plot routine
    ax = waterfall_lowlevel(fvals, ax, size)

    # apply options, if necessary
    if options is not None:
        options = VisualizationOptions(options)
        # apply_options(ax, options)
    else:
        ref = None

    # parse reference points
    if reference is not None:
        ref = []
        if isinstance(reference, list):
            for i_ref in reference:
                ref.append(ReferencePoint(i_ref))
        else:
            ref.append(ReferencePoint(reference))
    elif options is not None:
        ref = options.reference
    else:
        ref = None

    # plot reference points
    if ref is not None:
        ref_len = len(ref)
        for i_num, i_ref in enumerate(ref):
            ax.plot([0, len(fvals) - 1],
                    [i_ref.fval, i_ref.fval],
                    '--',
                    color=[0., 0.5 * (1. + i_num/ref_len), 0., 0.9])

    return ax


def waterfall_lowlevel(fvals, ax=None, size=(18.5, 10.5)):
    """
    Plot waterfall plot using list of function values.

    Parameters
    ----------

    fvals: numeric list or array
        Including values need to be plotted.

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        see waterfall

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # axes
    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # parse input
    fvals = np.array(fvals)

    n_fvals = len(fvals)
    start_ind = range(n_fvals)

    # assign colors
    # note: this has to happen before sorting
    # to get the same colors in different plots
    colors = assign_clustered_colors(fvals)

    # sort
    indices = sorted(range(n_fvals),
                     key=lambda j: fvals[j])

    # plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(start_ind, fvals)
    for j in range(n_fvals):
        j_fval = indices[j]
        color = colors[j_fval]
        fval = fvals[j_fval]
        ax.plot(j, fval, color=color, marker='o')

    # labels
    ax.set_xlabel('Ordered optimizer run')
    ax.set_ylabel('Function value')
    ax.set_title('Waterfall plot')

    return ax
