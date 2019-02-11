import numpy as np


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
                 reference=None,
                 x=None,
                 fval=None):
        super().__init__()

        if (reference is not None) and ((x is not None) or (fval is not None)):
            raise ("Please specify either an argument for reference or for x "
                   "and fval, but not both.")

        if isinstance(reference, dict) or isinstance(reference,
                                                     ReferencePoint):
            self.x = reference["x"]
            self.fval = reference["fval"]
        elif isinstance(reference, tuple):
            self.x = reference[1]
            self.fval = reference[2]

        if x is not None:
            self.x = np.array(x)
            self.fval = fval

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def create_references(references=None, x=None, fval=None):
    """
    This function handles the options, which are passed to the plotting
    routines

    Parameters
    ----------

    references: ReferencePoint or dict or list, optional
        Will be converted into a list of RefPoints

    x: ndarray, optional
        Parameter vector which should be used for reference point

    fval: float, optional
        Objective function value which should be used for reference point
    """

    # parse input (reference)
    ref = []
    if references is not None:
        if isinstance(references, list):
            for reference in references:
                ref.append(ReferencePoint(reference))
        else:
            ref.append(ReferencePoint(references))

    # parse input (x and fval)
    if (x is not None) and (fval is not None):
        ref.append(ReferencePoint(x=x, fval=fval))

    return ref
