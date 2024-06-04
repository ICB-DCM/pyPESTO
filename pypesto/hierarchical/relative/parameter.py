import logging

from ...C import InnerParameterType
from ..base_parameter import InnerParameter

logger = logging.getLogger(__name__)


class RelativeInnerParameter(InnerParameter):
    """A relative (inner) parameter of the relative hierarchical optimization problem.

    Attributes
    ----------
    observable_id:
        The id of the observable the relative inner parameter is connected to.
    group:
        Group index. Corresponds to `amici_model.index(observable_id)` + 1.
    """

    def __init__(
        self,
        *args,
        observable_id: str = None,
        group: int = None,
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

        self.observable_id = observable_id
        self.group = group
