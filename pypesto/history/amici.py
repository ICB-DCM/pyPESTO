import time
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from ..C import (
    FVAL,
    GRAD,
    HESS,
    N_ITERATIONS,
    RES,
    SRES,
    RDATAS,
    TIME,
    ModeType,
    X,
    CPU_TIME_TOTAL,
    PREEQ_CPU_TIME,
    POSTEQ_CPU_TIME_B,
    POSTEQ_CPU_TIME,
    PREEQ_CPU_TIME_B
)
from .base import add_fun_from_res, reduce_result_via_options
from .hdf5 import Hdf5History, with_h5_file
from .options import HistoryOptions
from .util import ResultDict, trace_wrap


class Hdf5AmiciHistory(Hdf5History):
    """
    Stores a representation of the history in an HDF5 file, extended with
    AMICI-specific traces of total simulation time, pre-equilibration time and
    post-equilibration time

    Parameters
    ----------
    id:
        Id of the history
    file:
        HDF5 file name.
    options:
        History options. Defaults to ``None``.
    """

    def __init__(
        self,
        id: str,
        file: Union[str, Path],
        options: Union[HistoryOptions, dict, None] = None,
    ):
        super().__init__(id, file, options=options)

    @staticmethod
    def _simulation_to_values(x, result, used_time):
        values = Hdf5History._simulation_to_values(x, result, used_time)
        values |= {
            CPU_TIME_TOTAL: sum([rdata[CPU_TIME_TOTAL] for rdata in
                                 result[RDATAS]]),
            PREEQ_CPU_TIME: sum([rdata[PREEQ_CPU_TIME] for rdata in
                                 result[RDATAS]]),
            PREEQ_CPU_TIME_B: sum([rdata[PREEQ_CPU_TIME_B] for rdata in
                                   result[RDATAS]]),
            POSTEQ_CPU_TIME: sum([rdata[POSTEQ_CPU_TIME] for rdata in
                                  result[RDATAS]]),
            POSTEQ_CPU_TIME_B: sum([rdata[POSTEQ_CPU_TIME_B] for rdata in
                                    result[RDATAS]])}
        return values

    @trace_wrap
    def get_cpu_time_total_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative simulation CPU time [ms].
        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return self._get_hdf5_entries(CPU_TIME_TOTAL, ix)

    @trace_wrap
    def get_preeq_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative computation time of the steady state solver [ms].
        (preequilibration)
        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return self._get_hdf5_entries(PREEQ_CPU_TIME, ix)

    @trace_wrap
    def get_preeq_timeB_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative computation time of the steady state solver of the backward
        problem [ms] (preequilibration).
        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return self._get_hdf5_entries(PREEQ_CPU_TIME_B, ix)

    @trace_wrap
    def get_posteq_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative computation time of the steady state solver [ms]
        (postequilibration).
        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return self._get_hdf5_entries(POSTEQ_CPU_TIME, ix)

    @trace_wrap
    def get_posteq_timeB_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative computation time of the steady state solver of the backward
        problem [ms] (postequilibration).
        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return self._get_hdf5_entries(POSTEQ_CPU_TIME_B, ix)

