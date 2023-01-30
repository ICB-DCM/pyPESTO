import logging

from ...C import InnerParameterType
from ..parameter import InnerParameter

logger = logging.getLogger(__name__)


class OptimalScalingParameter(InnerParameter):
    """A optimal scaling (inner) parameter of the optimal scaling hierarchical
    optimization problem.

    Attributes
    ----------
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
        *args,
        category: int = None,
        group: int = None,
        estimate: bool = False,
        **kwargs,
    ):
        """Construct.

        Parameters
        ----------
        See class attributes.
        """
        super().__init__(*args, **kwargs)
        if self.inner_parameter_type != InnerParameterType.OPTIMALSCALING:
            raise ValueError(
                "For the OptimalScalingParameter class, the parameter type has to be qualitative_scaling."
            )

        if group is None:
            raise ValueError("No Parameter group provided.")
        if category is None:
            raise ValueError("No Category provided.")

        self.category = category
        self.group = group
        self.estimate = estimate
        self.value = self.dummy_value
