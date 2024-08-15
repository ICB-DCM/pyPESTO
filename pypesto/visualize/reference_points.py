from collections.abc import Sequence
from typing import Optional, Union

import numpy as np


class ReferencePoint(dict):
    """
    Reference point for plotting.

    Should contain a parameter value and an objective function value,
    may also contain a color and a legend.

    Can be used like a dict.

    Attributes
    ----------
    x:
        Reference parameters.
    fval: float
        Function value, fun(x), for reference parameters.
    color: RGBA, optional
        Color which should be used for reference point.
    auto_color: boolean
        flag indicating whether color for this reference point should be
        assigned automatically or whether it was assigned by user
    legend: str
        legend text for reference point
    """

    def __init__(
        self,
        reference: Union[None, dict, tuple, "ReferencePoint"] = None,
        x: Optional[Sequence] = None,
        fval: Optional[float] = None,
        color=None,
        legend: Optional[str] = None,
    ):
        super().__init__()

        if (reference is not None) and ((x is not None) or (fval is not None)):
            raise ValueError(
                "Please specify either an argument for reference "
                "or for x and fval, but not both."
            )

        # assign legend, may be None
        self.legend = legend
        if isinstance(reference, dict) or isinstance(
            reference, ReferencePoint
        ):
            # Handle case of dict or ReferencePoint
            self.x = np.array(reference["x"])
            self.fval = reference["fval"]
            if "color" in reference.keys():
                self.color = reference["color"]
                if "auto_color" in reference.keys():
                    self.auto_color = reference["auto_color"]
                else:
                    self.auto_color = False
            else:
                self.color = None
                self.auto_color = True
            if "legend" in reference.keys():
                self.legend = reference["legend"]
            else:
                self.legend = None
        elif isinstance(reference, tuple):
            # Handle case of tuple
            self.x = np.array(reference[0])
            self.fval = reference[1]
            if len(reference) > 2:
                self.color = reference[2]
                self.auto_color = False
            else:
                self.color = None
                self.auto_color = True
        if reference is None:
            if x is not None:
                self.x = np.array(x)
            else:
                raise ValueError(
                    "Parameter vector x not passed, but is a "
                    "mandatory input when creating a reference "
                    "point. Stopping."
                )
            if fval is not None:
                self.fval = fval
            else:
                raise ValueError(
                    "Objective value fval not passed, but is a "
                    "mandatory input when creating a reference "
                    "point. Stopping."
                )
            if color is not None:
                self.color = color
                self.auto_color = False
            else:
                self.color = None
                self.auto_color = True

    def __getattr__(self, key):
        """Allow usage of keys as attributes."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def assign_colors(ref: Sequence[ReferencePoint]) -> Sequence[ReferencePoint]:
    """
    Assign colors to reference points, depending on user settings.

    Parameters
    ----------
    ref:
        Reference points, which need to get their color property filled

    Returns
    -------
    Reference points, which got their color property filled
    """
    # loop over reference points
    auto_color_count = 0
    for i_ref in ref:
        if i_ref["auto_color"]:
            auto_color_count += 1

    auto_colors = [
        [0.0, 0.5 * (1.0 + i_auto / auto_color_count), 0.0, 0.9]
        for i_auto in range(auto_color_count)
    ]

    # loop over reference points and assign auto_colors
    auto_color_count = 0
    for i_num, i_ref in enumerate(ref):
        if i_ref["auto_color"]:
            i_ref["color"] = auto_colors[i_num]
            auto_color_count += 1

    return ref


def create_references(
    references=None, x=None, fval=None, color=None, legend=None
) -> list[ReferencePoint]:
    """
    Create a list of reference point objects from user inputs.

    Parameters
    ----------
    references: ReferencePoint or dict or list, optional
        Will be converted into a list of RefPoints
    x: ndarray, optional
        Parameter vector which should be used for reference point
    fval: float, optional
        Objective function value which should be used for reference point
    color: RGBA, optional
        Color which should be used for reference point.
    legend: str
        legend text for reference point

    Returns
    -------
    colors: list of RGBA
        One for each element in 'vals'.
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
        ref.append(ReferencePoint(x=x, fval=fval, color=color, legend=legend))

    # assign colors for reference points which have no user-specified colors
    return assign_colors(ref)
