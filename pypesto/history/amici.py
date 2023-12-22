from pathlib import Path
from typing import Sequence, Union

import numpy as np

from ..C import (
    RDATAS,
    CPU_TIME_TOTAL,
    PREEQ_CPU_TIME,
    POSTEQ_CPU_TIME_B,
    POSTEQ_CPU_TIME,
    PREEQ_CPU_TIME_B
)
from .csv import CsvHistory
from .hdf5 import Hdf5History
from .options import HistoryOptions
from .util import trace_wrap


class Hdf5AmiciHistory(Hdf5History):
    """
    Stores a representation of the history in an HDF5 file, extended with
    AMICI-specific traces of total simulation time, pre-equilibration time and
    post-equilibration time.

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


class CsvAmiciHistory(CsvHistory):
    """Stores a representation of the history in a CSV file, extended with
    AMICI-specific traces of total simulation time, pre-equilibration time and
    post-equilibration time.

    Parameters
    ----------
    file:
        CSV file name.
    x_names:
        Parameter names.
    options:
        History options.
    load_from_file:
        If True, history will be initialized from data in the specified file.
    """

    def __init__(
        self,
        file: str,
        x_names: Sequence[str] = None,
        options: Union[HistoryOptions, dict] = None,
        load_from_file: bool = False,
    ):
        super().__init__(file, x_names, options, load_from_file=load_from_file)

    def _trace_columns(self) -> list[tuple]:
        columns = super()._trace_columns()
        return columns + [
            (c, np.nan)
            for c in [
                CPU_TIME_TOTAL,
                PREEQ_CPU_TIME,
                PREEQ_CPU_TIME_B,
                POSTEQ_CPU_TIME,
                POSTEQ_CPU_TIME_B
            ]
        ]

    def _simulation_to_values(self, result, used_time):
        values = super()._simulation_to_values(result, used_time)
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
        return list(self._trace[(CPU_TIME_TOTAL, np.nan)].values[ix])

    @trace_wrap
    def get_preeq_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative computation time of the steady state solver [ms].
        (pre-equilibration)
        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return list(self._trace[(PREEQ_CPU_TIME, np.nan)].values[ix])

    @trace_wrap
    def get_preeq_timeB_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative computation time of the steady state solver of the backward
        problem [ms] (pre-equilibration).
        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return list(self._trace[(PREEQ_CPU_TIME_B, np.nan)].values[ix])

    @trace_wrap
    def get_posteq_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative computation time of the steady state solver [ms]
        (post-equilibration).
        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return list(self._trace[(POSTEQ_CPU_TIME, np.nan)].values[ix])

    @trace_wrap
    def get_posteq_timeB_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative computation time of the steady state solver of the backward
        problem [ms] (post-equilibration).
        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return list(self._trace[(POSTEQ_CPU_TIME_B, np.nan)].values[ix])

