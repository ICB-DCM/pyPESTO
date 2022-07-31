import abc
import time
from typing import Dict, Sequence, Tuple, Union

"""Base history class."""

import numpy as np

from ..C import (
    CHI2,
    FVAL,
    GRAD,
    HESS,
    MODE_FUN,
    MODE_RES,
    RES,
    SCHI2,
    SRES,
    TIME,
    ModeType,
    X,
)
from ..util import (
    chi2_to_fval,
    res_to_chi2,
    schi2_to_grad,
    sres_to_fim,
    sres_to_schi2,
)
from .options import HistoryOptions
from .util import MaybeArray, ResultDict


class HistoryBase(abc.ABC):
    """Base class for history objects.

    Can be used as a dummy history, but does not implement any functionality.
    """

    # values calculated by the objective function
    RESULT_KEYS = (FVAL, GRAD, HESS, RES, SRES)
    # history also knows chi2, schi2
    FULL_RESULT_KEYS = (*RESULT_KEYS, CHI2, SCHI2)
    # all possible history entries
    ALL_KEYS = (X, *FULL_RESULT_KEYS, TIME)

    def __len__(self) -> int:
        """Define length by number of stored entries in the history."""
        raise NotImplementedError()

    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        """
        Update history after a function evaluation.

        Parameters
        ----------
        x:
            The parameter vector.
        sensi_orders:
            The sensitivity orders computed.
        mode:
            The objective function mode computed (function value or residuals).
        result:
            The objective function values for parameters `x`, sensitivities
            `sensi_orders` and mode `mode`.
        """

    def finalize(
        self,
        message: str = None,
        exitflag: str = None,
    ) -> None:
        """
        Finalize history. Called after a run. Default: Do nothing.

        Parameters
        ----------
        message:
            Optimizer message to be saved.
        exitflag:
            Optimizer exitflag to be saved.
        """

    @property
    def n_fval(self) -> int:
        """Return number of function evaluations."""
        raise NotImplementedError()

    @property
    def n_grad(self) -> int:
        """Return number of gradient evaluations."""
        raise NotImplementedError()

    @property
    def n_hess(self) -> int:
        """Return number of Hessian evaluations."""
        raise NotImplementedError()

    @property
    def n_res(self) -> int:
        """Return number of residual evaluations."""
        raise NotImplementedError()

    @property
    def n_sres(self) -> int:
        """Return number or residual sensitivity evaluations."""
        raise NotImplementedError()

    @property
    def start_time(self) -> float:
        """Return start time."""
        raise NotImplementedError()

    def get_x_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """
        Return parameters.

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        raise NotImplementedError()

    def get_fval_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[float], float]:
        """
        Return function values.

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        raise NotImplementedError()

    def get_grad_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """
        Return gradients.

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        raise NotImplementedError()

    def get_hess_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """
        Return hessians.

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        raise NotImplementedError()

    def get_res_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """
        Residuals.

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        raise NotImplementedError()

    def get_sres_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """
        Residual sensitivities.

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        raise NotImplementedError()

    def get_chi2_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[float], float]:
        """
        Chi2 values.

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        raise NotImplementedError()

    def get_schi2_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """
        Chi2 sensitivities.

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        raise NotImplementedError()

    def get_time_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[float], float]:
        """
        Cumulative execution times.

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        raise NotImplementedError()

    def get_trimmed_indices(self) -> np.ndarray:
        """Get indices for a monotonically decreasing history."""
        fval_trace = self.get_fval_trace()
        return np.where(fval_trace <= np.fmin.accumulate(fval_trace))[0]


class History(HistoryBase):
    """
    Tracks number of function evaluations only, no trace.

    Parameters
    ----------
    options:
        History options.
    """

    def __init__(self, options: Union[HistoryOptions, Dict] = None):
        self._n_fval: int = 0
        self._n_grad: int = 0
        self._n_hess: int = 0
        self._n_res: int = 0
        self._n_sres: int = 0
        self._start_time = time.time()

        if options is None:
            options = HistoryOptions()
        options = HistoryOptions.assert_instance(options)
        self.options: HistoryOptions = options

    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        """Update history after a function evaluation.

        Parameters
        ----------
        x:
            The parameter vector.
        sensi_orders:
            The sensitivity orders computed.
        mode:
            The objective function mode computed (function value or residuals).
        result:
            The objective function values for parameters `x`, sensitivities
            `sensi_orders` and mode `mode`.
        """
        self._update_counts(sensi_orders, mode)

    def _update_counts(
        self,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
    ):
        """Update the counters."""
        if mode == MODE_FUN:
            if 0 in sensi_orders:
                self._n_fval += 1
            if 1 in sensi_orders:
                self._n_grad += 1
            if 2 in sensi_orders:
                self._n_hess += 1
        elif mode == MODE_RES:
            if 0 in sensi_orders:
                self._n_res += 1
            if 1 in sensi_orders:
                self._n_sres += 1

    @property
    def n_fval(self) -> int:
        """See `HistoryBase` docstring."""
        return self._n_fval

    @property
    def n_grad(self) -> int:
        """See `HistoryBase` docstring."""
        return self._n_grad

    @property
    def n_hess(self) -> int:
        """See `HistoryBase` docstring."""
        return self._n_hess

    @property
    def n_res(self) -> int:
        """See `HistoryBase` docstring."""
        return self._n_res

    @property
    def n_sres(self) -> int:
        """See `HistoryBase` docstring."""
        return self._n_sres

    @property
    def start_time(self) -> float:
        """See `HistoryBase` docstring."""
        return self._start_time


def add_fun_from_res(result: ResultDict) -> ResultDict:
    """Calculate function values from residual values.

    Copies the result, but apart performs calculations only if entries
    are not present yet in the result object
    (thus can be called repeatedly).

    Parameters
    ----------
    result: Result dictionary from the objective function.

    Returns
    -------
    full_result:
        Result dicionary, adding whatever is possible to calculate.
    """
    result = result.copy()

    # calculate function values from residuals
    if result.get(CHI2) is None:
        result[CHI2] = res_to_chi2(result.get(RES))
    if result.get(SCHI2) is None:
        result[SCHI2] = sres_to_schi2(result.get(RES), result.get(SRES))
    if result.get(FVAL) is None:
        result[FVAL] = chi2_to_fval(result.get(CHI2))
    if result.get(GRAD) is None:
        result[GRAD] = schi2_to_grad(result.get(SCHI2))
    if result.get(HESS) is None:
        result[HESS] = sres_to_fim(result.get(SRES))

    return result


def reduce_result_via_options(
    result: ResultDict, options: HistoryOptions
) -> ResultDict:
    """Set values not to be stored in history or missing to NaN.

    Parameters
    ----------
    result:
        Result dictionary with all fields present.
    options:
        History options.

    Returns
    -------
    result:
        Result reduced to what is intended to be stored in history.
    """
    result = result.copy()

    # apply options to result
    for key in HistoryBase.FULL_RESULT_KEYS:
        if result.get(key) is None or not options.get(
            f'trace_record_{key}', True
        ):
            result[key] = np.nan

    return result
