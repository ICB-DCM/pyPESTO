"""Startpoint base classes."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ..C import FVAL, GRAD
from ..objective import NegLogParameterPriors, ObjectiveBase

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
        startpoints: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate startpoints.

        Parameters
        ----------
        n_starts: Number of starts.
        problem: Problem specifying e.g. dimensions, bounds, and guesses.
        startpoints:
            Explicit points to use as the first startpoints, shape
            ``(k, problem.dim)`` with ``k <= n_starts``. Any remaining
            startpoints are generated as usual.

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
        startpoints: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate a (n_starts, dim) nan matrix."""
        if startpoints is not None and len(startpoints) > 0:
            raise ValueError(
                "Explicit `startpoints` were provided, but this optimizer "
                "does not use startpoints."
            )
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

            .. deprecated::
                ``problem.x_guesses`` is deprecated. Pass explicit starting
                points via ``pypesto.optimize.minimize(...,
                startpoints=...)`` instead.
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
        startpoints: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate checked startpoints."""
        dim = problem.dim

        # shape: (k, dim)
        x_explicit = (
            np.zeros(shape=(0, dim))
            if startpoints is None
            else np.asarray(startpoints)
        )
        if x_explicit.size and x_explicit.shape[1] != dim:
            raise ValueError(
                f"`startpoints` must have shape (k, {dim}), got "
                f"{x_explicit.shape}."
            )

        # shape: (n_guesses, dim). `problem.x_guesses` is deprecated;
        # bypass the public property to avoid an extra warning on read,
        # and only warn if the deprecated guesses are actually used here.
        x_guesses = np.zeros(shape=(0, dim))
        if self.use_guesses:
            x_guesses = problem._x_guesses_full[:, problem.x_free_indices]
            if x_guesses.shape[0] > 0:
                warnings.warn(
                    "`problem.x_guesses` is deprecated and will be removed "
                    "in a future release. Pass explicit starting points via "
                    "`pypesto.optimize.minimize(..., startpoints=...)` "
                    "instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        x_have = np.vstack([x_explicit, x_guesses])
        lb, ub = problem.lb_init, problem.ub_init

        # number of required startpoints
        n_have = x_have.shape[0]
        n_required = n_starts - n_have

        if n_required <= 0:
            xs = x_have[:n_starts, :]
        else:
            # apply startpoint method
            x_sampled = self.sample(
                n_starts=n_required, lb=lb, ub=ub, priors=problem.x_priors
            )
            xs = np.vstack([x_have, x_sampled])

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
        priors: NegLogParameterPriors | None = None,
    ) -> np.ndarray:
        """Actually sample startpoints.

        While in this implementation, `__call__` handles the checking of
        guesses and resampling, this method defines the actual sampling.

        Parameters
        ----------
        n_starts: Number of startpoints to generate.
        lb: Lower parameter bound.
        ub: Upper parameter bound.
        priors: Parameter priors, if available. Some sampling methods may
            utilize this information.

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
        priors: NegLogParameterPriors | None = None,
    ) -> np.ndarray:
        """Check sampled points for fval, grad, and potentially resample ones.

        Parameters
        ----------
        xs: Startpoints candidates, shape (n_starts, n_par).
        lb: Lower parameter bound.
        ub: Upper parameter bound.
        objective: Objective function, for evaluation.
        priors: Parameter priors, if available.

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
                x = self.sample(n_starts=1, lb=lb, ub=ub, priors=priors)

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
        priors: NegLogParameterPriors | None = None,
    ) -> np.ndarray:
        """Call function."""
        return self.function(n_starts=n_starts, lb=lb, ub=ub, priors=priors)


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
