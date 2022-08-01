"""In-memory history."""

import time
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np

from ..C import CHI2, FVAL, GRAD, HESS, RES, SCHI2, SRES, TIME, ModeType, X
from .base import (
    History,
    HistoryBase,
    add_fun_from_res,
    reduce_result_via_options,
)
from .options import HistoryOptions
from .util import MaybeArray, ResultDict, trace_wrap


class MemoryHistory(History):
    """
    Class for optimization history stored in memory.

    Tracks number of function evaluations and keeps an in-memory
    trace of function evaluations.

    Parameters
    ----------
    options:
        History options.
    """

    def __init__(self, options: Union[HistoryOptions, Dict] = None):
        super().__init__(options=options)
        self._trace: Dict[str, Any] = {key: [] for key in HistoryBase.ALL_KEYS}

    def __len__(self) -> int:
        """Define length of history object."""
        return len(self._trace[TIME])

    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        """See `History` docstring."""
        super().update(x, sensi_orders, mode, result)
        self._update_trace(x, mode, result)

    def _update_trace(self, x, mode, result):
        """Update internal trace representation."""
        # calculating function values from residuals
        #  and reduce via requested history options
        result: dict = reduce_result_via_options(
            add_fun_from_res(result), self.options
        )

        result[X] = x

        used_time = time.time() - self._start_time
        result[TIME] = used_time

        for key in HistoryBase.ALL_KEYS:
            self._trace[key].append(result[key])

    @trace_wrap
    def get_x_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """See `HistoryBase` docstring."""
        return [self._trace[X][i] for i in ix]

    @trace_wrap
    def get_fval_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[float], float]:
        """See `HistoryBase` docstring."""
        return [self._trace[FVAL][i] for i in ix]

    @trace_wrap
    def get_grad_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return [self._trace[GRAD][i] for i in ix]

    @trace_wrap
    def get_hess_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return [self._trace[HESS][i] for i in ix]

    @trace_wrap
    def get_res_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return [self._trace[RES][i] for i in ix]

    @trace_wrap
    def get_sres_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return [self._trace[SRES][i] for i in ix]

    @trace_wrap
    def get_chi2_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[float], float]:
        """See `HistoryBase` docstring."""
        return [self._trace[CHI2][i] for i in ix]

    @trace_wrap
    def get_schi2_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return [self._trace[SCHI2][i] for i in ix]

    @trace_wrap
    def get_time_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[float], float]:
        """See `HistoryBase` docstring."""
        return [self._trace[TIME][i] for i in ix]
