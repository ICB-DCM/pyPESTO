import logging
from typing import Any, Literal, Optional

import numpy as np

from ..C import (
    DUMMY_INNER_VALUE,
    INNER_PARAMETER_BOUNDS,
    LIN,
    LOG,
    LOG10,
    LOWER_BOUND,
    UPPER_BOUND,
    InnerParameterType,
)

logger = logging.getLogger(__name__)


class InnerParameter:
    """
    An inner parameter of a hierarchical optimization problem.

    Attributes
    ----------
    coupled:
        If the inner parameter is part of an observable that has both
        an offset and scaling inner parameter, this attribute points to
        the other inner parameter. Otherwise, it is None.
    dummy_value:
        Value to be used when the optimal parameter is not yet known
        (in particular to simulate unscaled observables).
    inner_parameter_id:
        The inner parameter ID.
    inner_parameter_type:
        The inner parameter type.
    ixs:
        A mask (boolean matrix) that indicates the measurements that this
        parameter is used in.
    lb:
        The lower bound, for optimization.
    scale:
        Scale on which to estimate this parameter.
    ub:
        The upper bound, for optimization.
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
    ):
        """
        Construct.

        Parameters
        ----------
        See class attributes.
        """
        self.inner_parameter_id: str = inner_parameter_id
        self.coupled: InnerParameter = None
        self.inner_parameter_type: str = inner_parameter_type

        if scale not in {LIN, LOG, LOG10}:
            raise ValueError(f"Scale not recognized: {scale}.")

        if (
            scale in [LOG, LOG10]
            and inner_parameter_type == InnerParameterType.SIGMA
        ):
            raise ValueError(
                f"Inner parameter type `{inner_parameter_type}` "
                f"cannot be log-scaled."
            )

        if scale in [LOG, LOG10] and lb <= 0:
            raise ValueError(
                f"Lower bound of inner parameter `{inner_parameter_id}` "
                f"cannot be non-positive for log-scaled parameters. "
                f"Provide a positive lower bound."
            )

        self.scale = scale

        if inner_parameter_type not in (
            InnerParameterType.ORDINAL,
            InnerParameterType.OFFSET,
            InnerParameterType.SIGMA,
            InnerParameterType.SCALING,
            InnerParameterType.SPLINE,
        ):
            raise ValueError(
                f"Unsupported inner parameter type `{inner_parameter_type}`."
            )

        self.lb: float = lb
        self.ub: float = ub
        # Scaling and offset parameters can be bounded arbitrarily
        if inner_parameter_type not in (
            InnerParameterType.SCALING,
            InnerParameterType.OFFSET,
        ):
            self.check_bounds()
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

    def check_bounds(self):
        """Check bounds."""

        expected_lb = INNER_PARAMETER_BOUNDS[self.inner_parameter_type][
            LOWER_BOUND
        ]
        expected_ub = INNER_PARAMETER_BOUNDS[self.inner_parameter_type][
            UPPER_BOUND
        ]
        if self.lb != expected_lb or self.ub != expected_ub:
            raise ValueError(
                "Invalid bounds for inner parameters. Parameter ID: "
                f"`{self.inner_parameter_id}`. Provided bounds: "
                f"`[{self.lb}, {self.ub}]`. Expected bounds: "
                f"`[{expected_lb}, {expected_ub}]`. "
                f"All expected parameter bounds:\n{INNER_PARAMETER_BOUNDS}"
            )

    def is_within_bounds(self, value):
        """Check whether a value is within the bounds."""
        if value < self.lb or value > self.ub:
            return False
        return True

    def get_unsatisfied_bound(self, value) -> Optional[str]:
        """Get the unsatisfied bound index, if any."""
        if value < self.lb:
            return LOWER_BOUND
        elif value > self.ub:
            return UPPER_BOUND
        return None

    def get_bounds(self) -> dict:
        """Get the bounds."""
        return {LOWER_BOUND: self.lb, UPPER_BOUND: self.ub}
