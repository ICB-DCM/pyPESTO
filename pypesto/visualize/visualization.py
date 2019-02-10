import numpy as np


class VisualizationOptions(dict):
    """
    Options for visualization. Will contain things as color maps and options
    for the axes objects. At the moment, it contains only reference points for
    plotting.

    Can be used like a dict.

    Attributes
    ----------

    reference:
        list of reference points, which can be plotted together with results
    """

    def __init__(self,
                 reference=None):
        super().__init__()

        # initialize the list of reference points
        self.reference = []

        # if a list of possible references is passed, use the whole list
        if isinstance(reference, list):
            for i_ref in enumerate(reference):
                self.reference.append(ReferencePoint(i_ref))
        else:
            self.rerefence.append(ReferencePoint(reference))

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def handle_options(ax, options=None, reference=None):
    """
    This function handles the options, which are passed to the plotting
    routines

    Parameters
    ----------

    ax: matplotlib.Axes
        Axes object to use.

    options: VisualizationOptions, optional
        Options specifying axes, colors and reference points

    reference: list, optional
        List of reference points for optimization results, containing et
        least a function value fval
    """

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

    # return reference points, as they need to be applied seperately,
    # depending on the precise visualization routine
    return ref


class ReferencePoint(dict):
    """
    Reference point for plotting. Should contain a parameter value and an
    objective function value.

    Can be used like a dict.

    Attributes
    ----------

    x: ndarray
        Reference parameters.

    fval: float
        Function value, fun(x), for reference parameters.
    """

    def __init__(self,
                 x=None,
                 fval=None):
        super().__init__()

        if isinstance(x, dict) or isinstance(x, ReferencePoint):
            self.x = x["x"]
            self.fval = x["fval"]
        else:
            self.x = np.array(x)
            self.fval = fval

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
