"""Base history class."""

import numbers
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union

import numpy as np

from ..C import (
    FVAL,
    GRAD,
    HESS,
    MODE_FUN,
    MODE_RES,
    RES,
    SRES,
    TIME,
    ModeType,
    X,
)
from ..util import (
    fval_to_chi2,
    grad_to_schi2,
    res_to_fval,
    sres_to_fim,
    sres_to_grad,
)
from .options import HistoryOptions
from .util import MaybeArray, ResultDict


class HistoryBase(ABC):
    """Abstract base class for histories."""

    # values calculated by the objective function
    RESULT_KEYS = (FVAL, GRAD, HESS, RES, SRES)
    # all possible history entries
    ALL_KEYS = (X, *RESULT_KEYS, TIME)

    def __init__(self, options: Union[HistoryOptions, None] = None):
        if options is None:
            options = HistoryOptions()
        options = HistoryOptions.assert_instance(options)
        self.options: HistoryOptions = options

    @abstractmethod
    def update(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
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
        message: Union[str, None] = None,
        exitflag: Union[str, None] = None,
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

    @abstractmethod
    def __len__(self) -> int:
        """Define length by number of stored entries in the history."""

    @property
    @abstractmethod
    def n_fval(self) -> int:
        """Return number of function evaluations."""

    @property
    @abstractmethod
    def n_grad(self) -> int:
        """Return number of gradient evaluations."""

    @property
    @abstractmethod
    def n_hess(self) -> int:
        """Return number of Hessian evaluations."""

    @property
    @abstractmethod
    def n_res(self) -> int:
        """Return number of residual evaluations."""

    @property
    @abstractmethod
    def n_sres(self) -> int:
        """Return number or residual sensitivity evaluations."""

    @property
    @abstractmethod
    def start_time(self) -> float:
        """Return start time."""

    @property
    @abstractmethod
    def message(self) -> str:
        """Return message."""

    @property
    @abstractmethod
    def exitflag(self) -> str:
        """Return exitflag."""

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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
        fval_trace = self.get_fval_trace(ix, trim)
        reduce = isinstance(ix, numbers.Integral)
        if reduce:
            return fval_to_chi2(fval_trace)
        else:
            return [fval_to_chi2(fval) for fval in fval_trace]

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
        grad_trace = self.get_grad_trace(ix, trim)
        reduce = isinstance(ix, numbers.Integral)
        if reduce:
            return grad_to_schi2(grad_trace)
        else:
            return [grad_to_schi2(grad) for grad in grad_trace]

    @abstractmethod
    def get_time_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[float], float]:
        """
        Cumulative execution times [s].

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        raise NotImplementedError()

    def get_trimmed_indices(self) -> np.ndarray:
        """Get indices for a monotonically decreasing history."""
        fval_trace = self.get_fval_trace()
        return np.where(fval_trace <= np.fmin.accumulate(fval_trace))[0]

    def implements_trace(self) -> bool:
        """Check whether the history has a trace that can be queried."""
        try:
            self.get_fval_trace()
        except NotImplementedError:
            return False

        return True


class NoHistory(HistoryBase):
    """Dummy history that does not do anything.

    Can be used whenever a history object is needed, but no history is desired.
    Can be created, but not queried.
    """

    def update(  # noqa: D102
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        pass

    def __len__(self) -> int:  # noqa: D102
        raise NotImplementedError()

    @property
    def n_fval(self) -> int:  # noqa: D102
        raise NotImplementedError()

    @property
    def n_grad(self) -> int:  # noqa: D102
        raise NotImplementedError()

    @property
    def n_hess(self) -> int:  # noqa: D102
        raise NotImplementedError()

    @property
    def n_res(self) -> int:  # noqa: D102
        raise NotImplementedError()

    @property
    def n_sres(self) -> int:  # noqa: D102
        raise NotImplementedError()

    @property
    def start_time(self) -> float:  # noqa: D102
        raise NotImplementedError()

    @property
    def message(self) -> float:  # noqa: D102
        raise NotImplementedError()

    @property
    def exitflag(self) -> float:  # noqa: D102
        raise NotImplementedError()

    def get_x_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        raise NotImplementedError()

    def get_fval_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        raise NotImplementedError()

    def get_grad_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        raise NotImplementedError()

    def get_hess_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        raise NotImplementedError()

    def get_res_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        raise NotImplementedError()

    def get_sres_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        raise NotImplementedError()

    def get_time_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        raise NotImplementedError()


class CountHistoryBase(HistoryBase):
    """Abstract class tracking counts of function evaluations.

    Needs a separate implementation of trace.
    """

    def __init__(self, options: Union[HistoryOptions, dict] = None):
        super().__init__(options)
        self._n_fval: int = 0
        self._n_grad: int = 0
        self._n_hess: int = 0
        self._n_res: int = 0
        self._n_sres: int = 0
        self._start_time: float = time.time()
        self._exitflag = ""
        self._message = ""

    def update(  # noqa: D102
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        self._update_counts(sensi_orders, mode)

    def _update_counts(
        self,
        sensi_orders: tuple[int, ...],
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
    def n_fval(self) -> int:  # noqa: D102
        return self._n_fval

    @property
    def n_grad(self) -> int:  # noqa: D102
        return self._n_grad

    @property
    def n_hess(self) -> int:  # noqa: D102
        return self._n_hess

    @property
    def n_res(self) -> int:  # noqa: D102
        return self._n_res

    @property
    def n_sres(self) -> int:  # noqa: D102
        return self._n_sres

    @property
    def start_time(self) -> float:  # noqa: D102
        return self._start_time

    @property
    def message(self) -> str:  # noqa: D102
        return self._message

    @property
    def exitflag(self) -> str:  # noqa: D102
        return self._exitflag

    def finalize(  # noqa: D102
        self,
        message: str = None,
        exitflag: str = None,
    ) -> None:  # noqa: D102
        self._message = message
        self._exitflag = exitflag


class CountHistory(CountHistoryBase):
    """History that can only count, other functions cannot be invoked."""

    def __len__(self) -> int:  # noqa: D102
        raise NotImplementedError()

    def get_x_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        raise NotImplementedError()

    def get_fval_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        raise NotImplementedError()

    def get_grad_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        raise NotImplementedError()

    def get_hess_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        raise NotImplementedError()

    def get_res_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        raise NotImplementedError()

    def get_sres_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        raise NotImplementedError()

    def get_time_trace(  # noqa: D102
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        raise NotImplementedError()


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
    Result dictionary, adding whatever is possible to calculate.
    """
    result = result.copy()

    # calculate function values from residuals
    if result.get(FVAL) is None:
        result[FVAL] = res_to_fval(result.get(RES))
    if result.get(GRAD) is None:
        result[GRAD] = sres_to_grad(result.get(RES), result.get(SRES))
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
    Result reduced to what is intended to be stored in history.
    """
    result = result.copy()

    # apply options to result
    for key in HistoryBase.RESULT_KEYS:
        if result.get(key) is None or not options.get(
            f"trace_record_{key}", True
        ):
            result[key] = np.nan

    return result
