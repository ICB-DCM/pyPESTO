"""Startpoint base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..C import FVAL, GRAD
from ..objective import ObjectiveBase

if TYPE_CHECKING:
    import pypesto


class StartpointMethod(ABC):
    """Startpoint generation, in particular for multi-start optimization.

    Abstract base class, specific sampling method needs to be defined in
    sub-classes.
    """

    @abstractmethod
    def __call__(
        self,
        n_starts: int,
        problem: pypesto.problem.Problem,
    ) -> np.ndarray:
        """Generate startpoints.

        Parameters
        ----------
        n_starts: Number of starts.
        problem: Problem specifying e.g. dimensions, bounds, and guesses.

        Returns
        -------
        xs: Startpoints, shape (n_starts, n_par).
        """


class NoStartpoints(StartpointMethod):
    """Dummy class generating nan points. Useful if no startpoints needed."""

    def __call__(
        self,
        n_starts: int,
        problem: pypesto.problem.Problem,
    ) -> np.ndarray:
        """Generate a (n_starts, dim) nan matrix."""
        startpoints = np.full(shape=(n_starts, problem.dim), fill_value=np.nan)
        return startpoints


class CheckedStartpoints(StartpointMethod, ABC):
    """Startpoints checked for function value and/or gradient finiteness."""

    def __init__(
        self,
        use_guesses: bool = True,
        check_fval: bool = False,
        check_grad: bool = False,
    ):
        """Initialize.

        Parameters
        ----------
        use_guesses:
            Whether to use guesses provided in the problem.
        check_fval:
            Whether to check function values at the startpoint, and resample
            if not finite.
        check_grad:
            Whether to check gradients at the startpoint, and resample
            if not finite.
        """
        self.use_guesses: bool = use_guesses
        self.check_fval: bool = check_fval
        self.check_grad: bool = check_grad

    def __call__(
        self,
        n_starts: int,
        problem: pypesto.problem.Problem,
    ) -> np.ndarray:
        """Generate checked startpoints."""
        # shape: (n_guesses, dim)
        x_guesses = problem.x_guesses
        if not self.use_guesses:
            x_guesses = np.zeros(shape=(0, problem.dim))
        dim = problem.dim
        lb, ub = problem.lb_init, problem.ub_init

        # number of required startpoints
        n_guesses = x_guesses.shape[0]
        n_required = n_starts - n_guesses

        if n_required <= 0:
            return x_guesses[:n_starts, :]

        # apply startpoint method
        x_sampled = self.sample(n_starts=n_required, lb=lb, ub=ub)

        # assemble
        xs = np.zeros(shape=(n_starts, dim))
        xs[0:n_guesses, :] = x_guesses
        xs[n_guesses:n_starts, :] = x_sampled

        # check, resample and order startpoints
        xs = self.check_and_resample(
            xs=xs, lb=lb, ub=ub, objective=problem.objective
        )

        return xs

    @abstractmethod
    def sample(
        self,
        n_starts: int,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        """Actually sample startpoints.

        While in this implementation, `__call__` handles the checking of
        guesses and resampling, this method defines the actual sampling.

        Parameters
        ----------
        n_starts: Number of startpoints to generate.
        lb: Lower parameter bound.
        ub: Upper parameter bound.

        Returns
        -------
        xs: Startpoints, shape (n_starts, n_par).
        """

    def check_and_resample(
        self,
        xs: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        objective: ObjectiveBase,
    ) -> np.ndarray:
        """Check sampled points for fval, grad, and potentially resample ones.

        Parameters
        ----------
        xs: Startpoints candidates, shape (n_starts, n_par).
        lb: Lower parameter bound.
        ub: Upper parameter bound.
        objective: Objective function, for evaluation.

        Returns
        -------
        xs:
            Checked and potentially partially resampled startpoints,
            shape (n_starts, n_par).
        """
        if not self.check_fval and not self.check_grad:
            return xs

        if self.check_fval and not self.check_grad:
            sensi_orders = (0,)
        elif not self.check_fval and self.check_grad:
            sensi_orders = (1,)
        else:
            sensi_orders = 0, 1

        # track function values for ordering
        fvals = np.empty(shape=(xs.shape[0],))

        # iterate over all startpoint candidates
        for ix, x in enumerate(xs):
            # evaluate candidate
            objective.initialize()
            ret = objective(x, sensi_orders=sensi_orders, return_dict=True)
            fvals[ix] = ret.get(FVAL, np.nan)

            # loop until all requested sensis are finite
            while True:
                # discontinue if all requested sensis are finite
                if (0 not in sensi_orders or np.isfinite(ret[FVAL])) and (
                    1 not in sensi_orders or np.isfinite(ret[GRAD]).all()
                ):
                    break

                # resample a single point
                x = self.sample(n_starts=1, lb=lb, ub=ub)

                # evaluate candidate
                objective.initialize()
                ret = objective(x, sensi_orders=sensi_orders, return_dict=True)
                fvals[ix] = ret.get(FVAL, np.nan)

            # assign permissible value
            xs[ix] = x

        # sort startpoints by function value
        xs_order = np.argsort(fvals)
        xs = xs[xs_order, :]

        return xs


class FunctionStartpoints(CheckedStartpoints):
    """Define startpoints via callable.

    The callable should take the same arguments as the `__call__` method.
    """

    def __init__(
        self,
        function: Callable,
        use_guesses: bool = True,
        check_fval: bool = False,
        check_grad: bool = False,
    ):
        """Initialize.

        Parameters
        ----------
        function: The callable sampling startpoints.
        use_guesses, check_fval, check_grad: As in CheckedStartpoints.
        """
        super().__init__(
            use_guesses=use_guesses,
            check_fval=check_fval,
            check_grad=check_grad,
        )
        self.function: Callable = function

    def sample(
        self,
        n_starts: int,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        """Call function."""
        return self.function(n_starts=n_starts, lb=lb, ub=ub)


def to_startpoint_method(
    maybe_startpoint_method: StartpointMethod | Callable | bool,
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
    if maybe_startpoint_method is False:
        return NoStartpoints()
    raise TypeError(
        "Could not parse startpoint method of type "
        f"{type(maybe_startpoint_method)} to a StartpointMethod.",
    )
