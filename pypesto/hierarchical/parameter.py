import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


class InnerParameter:

    SCALING = 'scaling'
    OFFSET = 'offset'
    SIGMA = 'sigma'

    def __init__(self,
                 id: str,
                 type: str,
                 scale: str = 'lin',
                 lb: float = -np.inf,
                 ub: float = np.inf,
                 ixs: Any = None,
                 boring_val: float = None):
        """
        Parameters
        ----------
        id: str
            Id of the parameter.
        boring_val: float
            Value to be used when the parameter is not present (in particular
            to simulate unscaled observables).
        """
        self.id: str = id
        self.type: str = type

        if scale not in {'lin', 'log', 'log10'}:
            raise ValueError("Scale not recognized.")
        self.scale = scale

        self.lb = lb
        self.ub = ub
        self.ixs: Any = ixs

        if boring_val is None:
            if type == InnerParameter.SCALING:
                boring_val = 1.0
            elif type == InnerParameter.OFFSET:
                boring_val = 0.0
            elif type == InnerParameter.SIGMA:
                boring_val = 1.0
            else:
                raise ValueError("Could not deduce boring value for parameter "
                                 f"{id} of type {type}.")
        self.boring_val: float = boring_val
