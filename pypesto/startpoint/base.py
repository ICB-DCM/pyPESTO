"""Startpoint base classes."""

import numpy as np
from abc import abstractmethod
from typing import Callable, Union

from ..objective import ObjectiveBase


class StartpointMethod:
    """Startpoint generation, in particular for multi-start optimization.

    Abstract base class, specific sampling method needs to be defined in
    sub-classes.
    """

    @abstractmethod
    def __call__(
        self,
        n_starts: int,
        lb: np.ndarray,
        ub: np.ndarray,
        objective: ObjectiveBase,
    ) -> np.ndarray:
        """Generate startpoints.

        Parameters
        ----------
        n_starts:
            Number of starts.
        lb:
            Lower parameter bound.
        ub:
            Upper parameter bound.
        objective:
            Objective, maybe required for evaluation.

        Returns
        -------
        startpoints:
            Startpoints, shape (n_starts, n_par).
        """


class FunctionStartpoints(StartpointMethod):
    """Define startpoints via callable.

    The callable should take the same arguments as the `__call__` method.
    """

    def __init__(
        self,
        function: Callable,
    ):
        """
        Parameters
        ----------
        function: The callable sampling startpoints.
        """
        self.function = function

    def __call__(
        self,
        n_starts: int,
        lb: np.ndarray,
        ub: np.ndarray,
        objective: ObjectiveBase,
    ) -> np.ndarray:
        return self.function(
            n_starts=n_starts,
            lb=lb,
            ub=ub,
            objective=objective,
        )


def to_startpoint_method(
    maybe_startpoint_method: Union[StartpointMethod, Callable],
) -> StartpointMethod:
    """Create StartpointMethod instance if possible, otherwise raise.

    Parameters
    ----------
    maybe_startpoint_method:
        A StartpointMethod instance, or a Callable as expected by
        FunctionStartpoints.

    Returns
    -------
    startpoint_method:
        A StartpointMethod instance.

    Raises
    ------
    TypeError if arguments cannot be converted to a StartpointMethod.
    """
    if isinstance(maybe_startpoint_method, StartpointMethod):
        return maybe_startpoint_method
    if isinstance(maybe_startpoint_method, Callable):
        return FunctionStartpoints(maybe_startpoint_method)
    raise TypeError(
        "Could not parse startpoint method of type "
        f"{type(maybe_startpoint_method)} to a StartpointMethod.",
    )
