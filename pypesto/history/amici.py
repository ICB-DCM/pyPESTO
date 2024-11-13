from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np

from ..C import (
    CPU_TIME_TOTAL,
    POSTEQ_CPU_TIME,
    POSTEQ_CPU_TIME_BACKWARD,
    PREEQ_CPU_TIME,
    PREEQ_CPU_TIME_BACKWARD,
    RDATAS,
)
from .csv import CsvHistory
from .hdf5 import Hdf5History
from .options import HistoryOptions
from .util import trace_wrap


class Hdf5AmiciHistory(Hdf5History):
    """
    Stores history extended by AMICI-specific time traces in an HDF5 file.

    Stores AMICI-specific traces of total simulation time, pre-equilibration
    time and post-equilibration time.

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
    def load(
        id: str,
        file: Union[str, Path],
        options: Union[HistoryOptions, dict] = None,
    ) -> "Hdf5AmiciHistory":
        """Load the History object from memory."""
        history = Hdf5AmiciHistory(id=id, file=file, options=options)
        if options is None:
            history.recover_options(file)
        return history

    @staticmethod
    def _simulation_to_values(x, result, used_time):
        values = Hdf5History._simulation_to_values(x, result, used_time)
        # default unit for time in amici is [ms], converted to [s]
        values |= {
            key: sum([rdata[key] for rdata in result[RDATAS]]) * 0.001
            for key in (
                CPU_TIME_TOTAL,
                PREEQ_CPU_TIME,
                PREEQ_CPU_TIME_BACKWARD,
                POSTEQ_CPU_TIME,
                POSTEQ_CPU_TIME_BACKWARD,
            )
        }
        return values

    @trace_wrap
    def get_cpu_time_total_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative simulation CPU time [s].

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return self._get_hdf5_entries(CPU_TIME_TOTAL, ix)

    @trace_wrap
    def get_preeq_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative pre-equilibration time, [s].

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return self._get_hdf5_entries(PREEQ_CPU_TIME, ix)

    @trace_wrap
    def get_preeq_timeB_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative pre-equilibration time of the backward problem, [s].

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return self._get_hdf5_entries(PREEQ_CPU_TIME_BACKWARD, ix)

    @trace_wrap
    def get_posteq_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative post-equilibration time [s].

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return self._get_hdf5_entries(POSTEQ_CPU_TIME, ix)

    @trace_wrap
    def get_posteq_timeB_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative post-equilibration time of the backward problem [s].

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return self._get_hdf5_entries(POSTEQ_CPU_TIME_BACKWARD, ix)


class CsvAmiciHistory(CsvHistory):
    """
    Stores history extended by AMICI-specific time traces in a CSV file.

    Stores AMICI-specific traces of total simulation time, pre-equilibration
    time and post-equilibration time.

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
                PREEQ_CPU_TIME_BACKWARD,
                POSTEQ_CPU_TIME,
                POSTEQ_CPU_TIME_BACKWARD,
            ]
        ]

    def _simulation_to_values(self, result, used_time):
        values = super()._simulation_to_values(result, used_time)
        # default unit for time in amici is [ms], converted to [s]
        values |= {
            key: sum([rdata[key] for rdata in result[RDATAS]]) * 0.001
            for key in (
                CPU_TIME_TOTAL,
                PREEQ_CPU_TIME,
                PREEQ_CPU_TIME_BACKWARD,
                POSTEQ_CPU_TIME,
                POSTEQ_CPU_TIME_BACKWARD,
            )
        }
        return values

    @trace_wrap
    def get_cpu_time_total_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative simulation CPU time [s].

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return list(self._trace[(CPU_TIME_TOTAL, np.nan)].values[ix])

    @trace_wrap
    def get_preeq_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative pre-equilibration time [s].

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return list(self._trace[(PREEQ_CPU_TIME, np.nan)].values[ix])

    @trace_wrap
    def get_preeq_timeB_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative pre-equilibration time of the backward problem [s].

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return list(self._trace[(PREEQ_CPU_TIME_BACKWARD, np.nan)].values[ix])

    @trace_wrap
    def get_posteq_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative post-equilibration time [s].

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return list(self._trace[(POSTEQ_CPU_TIME, np.nan)].values[ix])

    @trace_wrap
    def get_posteq_timeB_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """
        Cumulative post-equilibration time of the backward problem [s].

        Takes as parameter an index or indices and returns corresponding trace
        values. If only a single value is requested, the list is flattened.
        """
        return list(self._trace[(POSTEQ_CPU_TIME_BACKWARD, np.nan)].values[ix])
