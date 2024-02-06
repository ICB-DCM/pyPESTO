"""Definition of an optimal scaling inner parameter class."""

import logging
from typing import Literal

from ...C import (
    INTERVAL_CENSORED,
    LEFT_CENSORED,
    RIGHT_CENSORED,
    InnerParameterType,
)
from ..base_parameter import InnerParameter

logger = logging.getLogger(__name__)


class OrdinalParameter(InnerParameter):
    """Inner parameter of the optimal scaling inner optimization problem for ordinal data.

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
    observable_id:
        The id of the observable whose measurements are in the category
        with this inner parameter.
    category:
        Category index.
    group:
        Group index.
    value:
        Current value of the inner parameter.
    estimate:
        Whether to estimate inner parameter in inner subproblem.
    censoring_type:
        The censoring type of the measurements in the category with this
        inner parameter. In case of ordinal measurements, this is None.
    """

    def __init__(
        self,
        *args,
        observable_id: str = None,
        group: int = None,
        category: int = None,
        estimate: bool = False,
        censoring_type: Literal[
            None, LEFT_CENSORED, RIGHT_CENSORED, INTERVAL_CENSORED
        ] = None,
        **kwargs,
    ):
        """Construct.

        Parameters
        ----------
        See class attributes.
        """
        super().__init__(*args, **kwargs)
        if self.inner_parameter_type != InnerParameterType.ORDINAL:
            raise ValueError(
                f"For the OptimalScalingParameter class, the parameter type has to be {InnerParameterType.ORDINAL}."
            )

        if group is None:
            raise ValueError("No Parameter group provided.")
        if category is None:
            raise ValueError("No Category provided.")

        self.observable_id = observable_id
        self.category = category
        self.group = group
        self.estimate = estimate
        self.value = self.dummy_value
        self.censoring_type = censoring_type

    def initialize(self):
        """Initialize."""
        if self.censoring_type is None:
            self.value = self.dummy_value
