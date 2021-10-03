"""Startpoint base classes."""

import numpy as np
from abc import abstractmethod
from typing import Callable

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
        x_guesses: np.ndarray,
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
        x_guesses:
            Externally provided guesses, shape (n_guess, n_par).
            Maybe used as reference points to generate remote points (e.g.
            maximizing some distance). If n_guesses >= n_starts, only the first
            n_starts guesses are returned.

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
            x_guesses: np.ndarray,
    ) -> np.ndarray:
        return self.function(
            n_starts=n_starts, lb=lb, ub=ub, objective=objective,
            x_guesses=x_guesses,
        )
