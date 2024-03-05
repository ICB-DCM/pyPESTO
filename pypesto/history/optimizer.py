"""Track optimal values during an optimization."""

import logging
from typing import Union

import numpy as np

from ..C import FVAL, GRAD, HESS, RES, SRES, ModeType, X
from ..util import allclose, is_none_or_nan, is_none_or_nan_array, isclose
from .base import HistoryBase, add_fun_from_res
from .util import ResultDict

logger = logging.getLogger(__name__)


class OptimizerHistory:
    """
    Optimizer objective call history.

    Container around a History object, additionally keeping track of optimal
    values.

    Attributes
    ----------
    fval0, fval_min:
        Initial and best function value found.
    x0, x_min:
        Initial and best parameters found.
    grad_min:
        gradient for best parameters
    hess_min:
        hessian (approximation) for best parameters
    res_min:
        residuals for best parameters
    sres_min:
        residual sensitivities for best parameters

    Parameters
    ----------
    history:
        History object to attach to this container. This history object
        implements the storage of the actual history.
    x0:
        Initial values for optimization.
    lb, ub:
        Lower and upper bound. Used for checking validity of optimal points.
    generate_from_history:
        If set to true, this function will try to fill attributes of this
        function based on the provided history. Defaults to ``False``.
    """

    # optimal point values
    MIN_KEYS = (X, *HistoryBase.RESULT_KEYS)

    def __init__(
        self,
        history: HistoryBase,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        generate_from_history: bool = False,
    ) -> None:
        self.history: HistoryBase = history

        # initial point
        self.fval0: Union[float, None] = None
        self.x0: np.ndarray = x0

        # bounds
        self.lb: np.ndarray = lb
        self.ub: np.ndarray = ub

        # minimum point
        self.fval_min: float = np.inf
        self.x_min: Union[np.ndarray, None] = None
        self.grad_min: Union[np.ndarray, None] = None
        self.hess_min: Union[np.ndarray, None] = None
        self.res_min: Union[np.ndarray, None] = None
        self.sres_min: Union[np.ndarray, None] = None

        if generate_from_history:
            self._maybe_compute_init_and_min_vals_from_trace()

    def update(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        """Update history and best found value.

        Parameters
        ----------
        x:
            Current parameter vector.
        sensi_orders:
            Sensitivity orders to be evaluated.
        mode:
            Mode of the evaluation.
        result:
            Current result.
        """
        result = add_fun_from_res(result)
        self._update_vals(x, result)
        self.history.update(x, sensi_orders, mode, result)

    def finalize(
        self,
        message: Union[str, None] = None,
        exitflag: Union[int, None] = None,
    ):
        """
        Finalize history.

        Parameters
        ----------
        message:
            Optimizer message to be saved. Defaults to ``None``.
        exitflag:
            Optimizer exitflag to be saved. Defaults to ``None``.
        """
        self.history.finalize(message=message, exitflag=exitflag)

        # There can be entries in the history e.g. for grad that are not
        #  recorded in ..._min, e.g. when evaluated before fval.
        # On the other hand, not all variables may be recorded in the history.
        # Thus, here at the end we go over the history once and try to fill
        #  in what is available.

        # check if a useful history exists
        # TODO Y This can be solved prettier
        try:
            self.history.get_x_trace()
        except NotImplementedError:
            return

        # find optimal point
        result = self._get_optimal_point_from_history()

        fval = result[FVAL]
        if fval is None:
            # nothing to be improved
            return

        # check if history has a better point (should not really happen)
        if (
            fval < self.fval_min
            and not isclose(fval, self.fval_min)
            and not allclose(result[X], self.x_min)
        ):
            # issue a warning, as if this happens, then something may be wrong
            logger.warning(
                f"History has a better point {fval} than the current best "
                f"point {self.fval_min}."
            )
            # update everything
            for key in self.MIN_KEYS:
                setattr(self, key + "_min", result[key])

        # check if history has same point
        if (
            isclose(fval, self.fval_min)
            and self.x_min is not None
            and allclose(result[X], self.x_min)
        ):
            # update only missing entries
            #  (e.g. grad and hess may be recorded but not in history)
            for key in self.MIN_KEYS:
                if result[key] is not None:
                    # if getattr(self, f'{key}_min') is None:
                    setattr(self, f"{key}_min", result[key])

    def _update_vals(self, x: np.ndarray, result: ResultDict) -> None:
        """Update initial and best function values."""
        # update initial point
        if is_none_or_nan(self.fval0) and np.array_equal(x, self.x0):
            self.fval0 = result.get(FVAL)

        # don't update optimal point if point is not admissible
        if not self._admissible(x):
            return

        # update if fval is better
        if (
            not is_none_or_nan(fval := result.get(FVAL))
            and fval < self.fval_min
        ):
            # need to update all values, as better fval found
            for key in HistoryBase.RESULT_KEYS:
                setattr(self, f"{key}_min", result.get(key))
            self.x_min = x
            return

        # Sometimes sensitivities are evaluated on subsequent calls. We can
        # identify this situation by checking that x hasn't changed.
        if self.x_min is not None and np.array_equal(self.x_min, x):
            for key in (GRAD, HESS, SRES):
                val_min = getattr(self, f"{key}_min", None)
                if is_none_or_nan_array(val_min) and not is_none_or_nan_array(
                    val := result.get(key)
                ):
                    setattr(self, f"{key}_min", val)

    def _maybe_compute_init_and_min_vals_from_trace(self) -> None:
        """Try to set initial and best function value from trace.

        .. note:: Only possible if history has a trace.
        """
        if not len(self.history):
            # nothing to be computed from empty history
            return

        # some optimizers may evaluate hess+grad first to compute trust region
        # etc
        for ix in range(len(self.history)):
            fval = self.history.get_fval_trace(ix)
            x = self.history.get_x_trace(ix)
            if not is_none_or_nan(fval) and allclose(x, self.x0):
                self.fval0 = float(fval)
                break

        # find best fval
        result = self._get_optimal_point_from_history()

        # assign values
        for key in OptimizerHistory.MIN_KEYS:
            setattr(self, f"{key}_min", result[key])

    def _admissible(self, x: np.ndarray) -> bool:
        """Check whether point `x` is admissible (i.e. within bounds).

        Parameters
        ----------
        x: A single parameter vector.

        Returns
        -------
        Whether the point fulfills the problem requirements.
        """
        return np.all(x <= self.ub) and np.all(x >= self.lb)

    def _get_optimal_point_from_history(self) -> ResultDict:
        """Extract optimal point from `self.history`."""
        result = {}

        # get indices of admissible trace entries
        # shape (n_sample, n_x)
        xs = np.asarray(self.history.get_x_trace())
        ixs_admit = [ix for ix, x in enumerate(xs) if self._admissible(x)]

        if len(ixs_admit) == 0:
            # no admittable indices
            return {key: None for key in OptimizerHistory.MIN_KEYS}

        # index of minimum of fval values
        ix_min = np.nanargmin(self.history.get_fval_trace(ixs_admit))
        # np.argmin returns ndarray when multiple minimal values are found,
        #  we want the first occurrence
        if isinstance(ix_min, np.ndarray):
            ix_min = ix_min[0]
        # select index in original array
        ix_min = ixs_admit[ix_min]

        # fill in parameter and function value from that index
        for var in (X, FVAL, RES):
            val = getattr(self.history, f"get_{var}_trace")(ix_min)
            if val is not None and not np.all(np.isnan(val)):
                result[var] = val
            # convert to float if var is FVAL to be sure
            if var == FVAL:
                result[var] = float(result[var])

        # derivatives may be evaluated at different indices, therefore
        #  iterate over all and check whether any has the same parameter
        #  and the desired field filled
        for var in (GRAD, HESS, SRES):
            for ix in range(len(self.history)):
                if not allclose(result[X], self.history.get_x_trace(ix)):
                    # different parameter
                    continue
                val = getattr(self.history, f"get_{var}_trace")(ix)
                if not is_none_or_nan_array(val):
                    result[var] = val
                    # successfuly found
                    break

        # fill remaining keys with None
        for key in OptimizerHistory.MIN_KEYS:
            if key not in result:
                result[key] = None

        return result
