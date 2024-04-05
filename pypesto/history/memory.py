"""In-memory history."""

import time
from collections.abc import Sequence
from typing import Any, Union

import numpy as np

from ..C import FVAL, GRAD, HESS, RES, SRES, TIME, ModeType, X
from .base import (
    CountHistoryBase,
    HistoryBase,
    add_fun_from_res,
    reduce_result_via_options,
)
from .options import HistoryOptions
from .util import MaybeArray, ResultDict, trace_wrap


class MemoryHistory(CountHistoryBase):
    """
    Class for optimization history stored in memory.

    Tracks number of function evaluations and keeps an in-memory
    trace of function evaluations.

    Parameters
    ----------
    options:
        History options, see :class:`pypesto.history.HistoryOptions`. Defaults
        to `None`, which implies default options.
    """

    def __init__(self, options: Union[HistoryOptions, dict, None] = None):
        super().__init__(options=options)
        self._trace: dict[str, Any] = {key: [] for key in HistoryBase.ALL_KEYS}

    def update(
        self,
        x: np.ndarray,
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        """See :meth:`HistoryBase.update`."""
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

    def __len__(self) -> int:
        """Define length of history object."""
        return len(self._trace[TIME])

    @trace_wrap
    def get_x_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """See :meth:`HistoryBase.get_x_trace`."""
        return [self._trace[X][i] for i in ix]

    @trace_wrap
    def get_fval_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[float], float]:
        """See :meth:`HistoryBase.get_fval_trace`."""
        return [self._trace[FVAL][i] for i in ix]

    @trace_wrap
    def get_grad_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See :meth:`HistoryBase.get_grad_trace`."""
        return [self._trace[GRAD][i] for i in ix]

    @trace_wrap
    def get_hess_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See :meth:`HistoryBase.get_hess_trace`."""
        return [self._trace[HESS][i] for i in ix]

    @trace_wrap
    def get_res_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See :meth:`HistoryBase.get_res_trace`."""
        return [self._trace[RES][i] for i in ix]

    @trace_wrap
    def get_sres_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See :meth:`HistoryBase.get_sres_trace`."""
        return [self._trace[SRES][i] for i in ix]

    @trace_wrap
    def get_time_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[float], float]:
        """See :meth:`HistoryBase.get_time_trace`."""
        return [self._trace[TIME][i] for i in ix]
