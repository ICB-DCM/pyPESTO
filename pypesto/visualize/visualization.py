import numpy as np


class VisualizeOptions(dict):
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

        self.x = x
        self.fval = fval

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
