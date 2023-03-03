"""Definition of an optimal scaling inner parameter class."""
import logging
from typing import Any, Literal

import numpy as np

from ...C import LIN, LOG, LOG10, InnerParameterType
from ..parameter import InnerParameter

logger = logging.getLogger(__name__)


class OptimalScalingParameter(InnerParameter):
    """Inner parameter of the optimal scaling hierarchical optimization problem.

    Attributes
    ----------
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
    category:
        Category index.
    group:
        Group index.
    value:
        Current value of the inner parameter.
    estimate:
        Whether to estimate inner parameter in inner subproblem.
    """

    def __init__(
        self,
        inner_parameter_id: str,
        inner_parameter_type: InnerParameterType,
        category: int,
        group: int,
        scale: Literal[LIN, LOG, LOG10] = LIN,
        lb: float = -np.inf,
        ub: float = np.inf,
        ixs: Any = None,
        dummy_value: float = None,
        estimate: bool = False,
    ):
        """Construct.

        Parameters
        ----------
        See class attributes.
        """
        super().__init__(
            inner_parameter_id=inner_parameter_id,
            inner_parameter_type=inner_parameter_type,
            scale=scale,
            lb=lb,
            ub=ub,
            ixs=ixs,
            dummy_value=dummy_value,
        )
        if self.inner_parameter_type != InnerParameterType.OPTIMAL_SCALING:
            raise ValueError(
                f"For the OptimalScalingParameter class, the parameter type has to be {InnerParameterType.OPTIMAL_SCALING}."
            )

        if group is None:
            raise ValueError("No Parameter group provided.")
        if category is None:
            raise ValueError("No Category provided.")

        self.category = category
        self.group = group
        self.estimate = estimate
        self.value = self.dummy_value
