import logging
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


class InnerParameter:
    """
    An inner parameter of a hierarchical optimization problem.

    Attributes
    ----------
    coupled:
        TODO
    """

    # Supported parameter types:
    SCALING = 'scaling'
    OFFSET = 'offset'
    SIGMA = 'sigma'
    OPTIMALSCALING = 'qualitative_scaling'

    def __init__(
        self,
        id: str,
        type: Literal['scaling', 'offset', 'sigma', 'qualitative_scaling'],
        scale: Literal['lin', 'log', 'log10'] = 'lin',
        lb: float = -np.inf,
        ub: float = np.inf,
        ixs: Any = None,
        boring_val: float = None,
        category: int = None,
        group: int = None,
    ):
        """
        Construct.

        Parameters
        ----------
        id:
            Id of the parameter.
        type:
            Type of this inner parameter.
        scale:
            Scale on which to estimate this parameter.
        lb:
            Lower bound for this parameter.
        ub:
            Upper bound for this parameter.
        ixs:
            TODO
        boring_val:
            Value to be used when the parameter is not present (in particular
            to simulate unscaled observables).
        category:
            TODO
            Only relevant if ``type==qualitative_scaling``.
        group:
            TODO
            Only relevant if ``type==qualitative_scaling``.
        """
        self.id: str = id
        self.coupled = False
        self.type: str = type

        if scale not in {'lin', 'log', 'log10'}:
            raise ValueError("Scale not recognized.")
        self.scale = scale

        if type not in (
            InnerParameter.OPTIMALSCALING,
            InnerParameter.OFFSET,
            InnerParameter.SIGMA,
            InnerParameter.SCALING,
        ):
            raise ValueError(f"Unsupported inner parameter type `{type}`.")

        if type == InnerParameter.OPTIMALSCALING:
            if group is None:
                raise ValueError("No Parameter group provided.")
            if category is None:
                raise ValueError("No Category provided.")
        self.group = group
        self.category = category

        self.lb: float = lb
        self.ub: float = ub
        self.ixs: Any = ixs

        if boring_val is None:
            if type == InnerParameter.SCALING:
                boring_val = 1.0
            elif type == InnerParameter.OFFSET:
                boring_val = 0.0
            elif type == InnerParameter.SIGMA:
                boring_val = 1.0
            elif type == InnerParameter.OPTIMALSCALING:
                boring_val = category
            else:
                raise ValueError(
                    "Could not deduce boring value for parameter "
                    f"{id} of type {type}."
                )
        self.boring_val: float = boring_val
