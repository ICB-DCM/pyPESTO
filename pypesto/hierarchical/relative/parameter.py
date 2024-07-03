import logging

from ...C import InnerParameterType
from ..base_parameter import InnerParameter

logger = logging.getLogger(__name__)


class RelativeInnerParameter(InnerParameter):
    """A relative (inner) parameter of the relative hierarchical optimization problem.

    Attributes
    ----------
    observable_ids:
        A list of IDs of the observables that the relative inner parameter is connected to.
        Can be a single or multiple observables.
    observable_indices:
        A list of indices of the observables that the relative inner parameter is connected to.
    """

    def __init__(
        self,
        *args,
        observable_ids: list[str] = None,
        observable_indices: list[int] = None,
        **kwargs,
    ):
        """Construct.

        Parameters
        ----------
        See class attributes.
        """
        super().__init__(*args, **kwargs)
        if self.inner_parameter_type not in [
            InnerParameterType.SCALING,
            InnerParameterType.OFFSET,
            InnerParameterType.SIGMA,
        ]:
            raise ValueError(
                "For the RelativeParameter class, the parameter type has to be scaling, offset or sigma."
            )

        self.observable_ids = observable_ids
        self.observable_indices = observable_indices
