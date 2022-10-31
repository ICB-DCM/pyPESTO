import logging
from typing import Any, Literal

import numpy as np

from ..C import DUMMY_INNER_VALUE, LIN, LOG, LOG10, InnerParameterType

logger = logging.getLogger(__name__)


class InnerParameter:
    """
    An inner parameter of a hierarchical optimization problem.

    Attributes
    ----------
    coupled:
        TODO
    """

    def __init__(
        self,
        inner_parameter_id: str,
        inner_parameter_type: InnerParameterType,
        scale: Literal[LIN, LOG, LOG10] = LIN,
        lb: float = -np.inf,
        ub: float = np.inf,
        ixs: Any = None,
        dummy_value: float = None,
        category: int = None,
        group: int = None,
    ):
        """
        Construct.

        Parameters
        ----------
        inner_parameter_id:
            Id of the parameter.
        inner_parameter_type:
            Type of this inner parameter.
        scale:
            Scale on which to estimate this parameter.
        lb:
            Lower bound for this parameter.
        ub:
            Upper bound for this parameter.
        ixs:
            Boolean matrix, indicating for which measurements this parameter
            is used.
        dummy_value:
            Value to be used when the optimal parameter is not yet known
            (in particular to simulate unscaled observables).
        category:
            Category index.
            Only relevant if ``type==qualitative_scaling``.
        group:
            Group index.
            Only relevant if ``type==qualitative_scaling``.
        """
        self.inner_parameter_id: str = inner_parameter_id
        self.coupled = False
        self.inner_parameter_type: str = inner_parameter_type

        if scale not in {LIN, LOG, LOG10}:
            raise ValueError(f"Scale not recognized: {scale}.")
        self.scale = scale

        if inner_parameter_type not in (
            # InnerParameterType.OPTIMALSCALING,
            InnerParameterType.OFFSET,
            InnerParameterType.SIGMA,
            InnerParameterType.SCALING,
        ):
            raise ValueError(
                f"Unsupported inner parameter type `{inner_parameter_type}`."
            )

        # if inner_parameter_type == InnerParameter.OPTIMALSCALING:
        #     if group is None:
        #         raise ValueError("No Parameter group provided.")
        #     if category is None:
        #         raise ValueError("No Category provided.")
        # self.group = group
        # self.category = category

        self.lb: float = lb
        self.ub: float = ub
        self.ixs: Any = ixs

        if dummy_value is None:
            try:
                dummy_value = DUMMY_INNER_VALUE[inner_parameter_type]
            except KeyError as e:
                raise ValueError(
                    "Unsupported parameter type. Parameter id:"
                    f"`{inner_parameter_id}`. Parameter type:"
                    f"`{inner_parameter_type}`."
                ) from e

        self.dummy_value: float = dummy_value
