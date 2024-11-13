import logging

from ...C import InnerParameterType
from ..base_parameter import InnerParameter

logger = logging.getLogger(__name__)


class SplineInnerParameter(InnerParameter):
    """A spline (inner) parameter of the spline hierarchical optimization problem.

    Attributes
    ----------
    observable_id:
        The id of the observable the spline is modeling.
    group:
        Group index. Corresponds to observable index + 1.
    index:
        Parameter index inside the group. Ranges from 1 to n_spline_parameters
        of its group.
    value:
        Current value of the inner parameter.
    estimate:
        Whether to estimate inner parameter in inner subproblem.
    """

    def __init__(
        self,
        *args,
        observable_id: str = None,
        group: int = None,
        index: int = None,
        estimate: bool = False,
        **kwargs,
    ):
        """Construct.

        Parameters
        ----------
        See class attributes.
        """
        super().__init__(*args, **kwargs)
        if self.inner_parameter_type not in [
            InnerParameterType.SPLINE,
            InnerParameterType.SIGMA,
        ]:
            raise ValueError(
                "For the SplineParameter class, the parameter type has to be spline or sigma."
            )

        if observable_id is None:
            raise ValueError("No observable id provided.")
        if group is None:
            raise ValueError("No Parameter group provided.")
        if (
            index is None
            and self.inner_parameter_type == InnerParameterType.SPLINE
        ):
            raise ValueError(
                "No Parameter index provided for spline parameter."
            )

        self.observable_id = observable_id
        self.group = group
        self.index = index
        self.estimate = estimate
        self.value = self.dummy_value

    def initialize(self):
        """Initialize."""
        self.value = self.dummy_value
