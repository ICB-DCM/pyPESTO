"""CSV history."""

import copy
import os
import time
from collections.abc import Sequence
from typing import Union

import numpy as np
import pandas as pd

from ..C import (
    FVAL,
    GRAD,
    HESS,
    N_FVAL,
    N_GRAD,
    N_HESS,
    N_RES,
    N_SRES,
    RES,
    SRES,
    TIME,
    ModeType,
    X,
)
from .base import CountHistoryBase, add_fun_from_res, reduce_result_via_options
from .options import HistoryOptions
from .util import MaybeArray, ResultDict, trace_wrap


class CsvHistory(CountHistoryBase):
    """Stores a representation of the history in a CSV file.

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
        super().__init__(options=options)
        self.x_names: Sequence[str] = x_names
        self._trace: Union[pd.DataFrame, None] = None
        self.file: str = os.path.abspath(file)

        # create trace file dirs
        if self.file is not None:
            dirname = os.path.dirname(self.file)
            os.makedirs(dirname, exist_ok=True)

        if load_from_file and os.path.exists(self.file):
            trace = pd.read_csv(self.file, header=[0, 1], index_col=0)
            # replace 'nan' in cols with np.NAN
            cols = pd.DataFrame(trace.columns.to_list())
            cols[cols == "nan"] = np.nan
            trace.columns = pd.MultiIndex.from_tuples(
                cols.to_records(index=False).tolist()
            )
            for col in trace.columns:
                # transform strings to np.ndarrays
                trace[col] = trace[col].apply(string2ndarray)

            self._trace = trace
            self.x_names = trace[X].columns
            self._update_counts_from_trace()

    def _update_counts_from_trace(self) -> None:
        self._n_fval = self._trace[(N_FVAL, np.nan)].max()
        self._n_grad = self._trace[(N_GRAD, np.nan)].max()
        self._n_hess = self._trace[(N_HESS, np.nan)].max()
        self._n_res = self._trace[(N_RES, np.nan)].max()
        self._n_sres = self._trace[(N_SRES, np.nan)].max()

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

    def finalize(self, message: str = None, exitflag: str = None):
        """See :meth:`HistoryBase.finalize`."""
        super().finalize(message=message, exitflag=exitflag)
        self._save_trace(finalize=True)

    def _simulation_to_values(self, result, used_time):
        values = {
            TIME: used_time,
            N_FVAL: self._n_fval,
            N_GRAD: self._n_grad,
            N_HESS: self._n_hess,
            N_RES: self._n_res,
            N_SRES: self._n_sres,
            FVAL: result[FVAL],
            RES: result[RES],
            HESS: result[HESS],
        }
        return values

    def _update_trace(
        self,
        x: np.ndarray,
        mode: ModeType,
        result: ResultDict,
    ):
        """Update and possibly store the trace."""
        if not self.options.trace_record:
            return

        # init trace
        if self._trace is None:
            self._init_trace(x)

        # calculating function values from residuals
        #  and reduce via requested history options
        result = reduce_result_via_options(
            add_fun_from_res(result), self.options
        )

        used_time = time.time() - self._start_time

        # create table row
        row = pd.Series(
            name=len(self._trace), index=self._trace.columns, dtype="object"
        )

        values = self._simulation_to_values(result, used_time)

        for var, val in values.items():
            row[(var, np.nan)] = val

        for var, val in {
            X: x,
            GRAD: result[GRAD],
        }.items():
            if var == X or self.options[f"trace_record_{var}"]:
                row[var] = val
            else:
                row[(var, np.nan)] = np.nan

        self._trace = pd.concat(
            (self._trace, pd.DataFrame([row])),
        )

        # save trace to file
        self._save_trace()

    def _trace_columns(self) -> list[tuple]:
        return [
            (c, np.nan)
            for c in [
                TIME,
                N_FVAL,
                N_GRAD,
                N_HESS,
                N_RES,
                N_SRES,
                FVAL,
                RES,
                SRES,
                HESS,
            ]
        ]

    def _init_trace(self, x: np.ndarray):
        """Initialize the trace."""
        if self.x_names is None:
            self.x_names = [f"x{i}" for i, _ in enumerate(x)]

        columns = self._trace_columns()

        for var in [X, GRAD]:
            if var == X or self.options[f"trace_record_{var}"]:
                columns.extend([(var, x_name) for x_name in self.x_names])
            else:
                columns.extend([(var,)])

        # TODO: multi-index for res, sres, hess
        self._trace = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(columns), dtype="float64"
        )

        # only non-float64
        trace_dtypes = {
            RES: "object",
            SRES: "object",
            HESS: "object",
            N_FVAL: "int64",
            N_GRAD: "int64",
            N_HESS: "int64",
            N_RES: "int64",
            N_SRES: "int64",
        }

        for var, dtype in trace_dtypes.items():
            self._trace[(var, np.nan)] = self._trace[(var, np.nan)].astype(
                dtype
            )

    def _save_trace(self, finalize: bool = False):
        """
        Save to file via :meth:`pandas.DataFrame.to_csv`.

        Only done, if `self.storage_file` is not None and other conditions.
        apply.
        """
        if self.file is None:
            return

        if finalize or (
            len(self._trace) > 0
            and len(self._trace) % self.options.trace_save_iter == 0
        ):
            # save
            trace_copy = copy.deepcopy(self._trace)
            for field in [(HESS, np.nan), (RES, np.nan), (SRES, np.nan)]:
                trace_copy[field] = trace_copy[field].apply(
                    ndarray2string_full
                )
            trace_copy.to_csv(self.file)

    def __len__(self) -> int:
        """Define length of history object."""
        return len(self._trace)

    @trace_wrap
    def get_x_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """See :meth:`HistoryBase.get_x_trace`."""
        return list(self._trace[X].values[ix])

    @trace_wrap
    def get_fval_trace(
        self, ix: Union[int, Sequence[int], None], trim: bool = False
    ) -> Union[Sequence[float], float]:
        """See :meth:`HistoryBase.get_fval_trace`."""
        return list(self._trace[(FVAL, np.nan)].values[ix])

    @trace_wrap
    def get_grad_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See :meth:`HistoryBase.get_grad_trace`."""
        return list(self._trace[GRAD].values[ix])

    @trace_wrap
    def get_hess_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See :meth:`HistoryBase.get_hess_trace`."""
        return list(self._trace[(HESS, np.nan)].values[ix])

    @trace_wrap
    def get_res_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See :meth:`HistoryBase.get_res_trace`."""
        return list(self._trace[(RES, np.nan)].values[ix])

    @trace_wrap
    def get_sres_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See :meth:`HistoryBase.get_sres_trace`."""
        return list(self._trace[(SRES, np.nan)].values[ix])

    @trace_wrap
    def get_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """See :meth:`HistoryBase.get_time_trace`."""
        return list(self._trace[(TIME, np.nan)].values[ix])


def ndarray2string_full(x: Union[np.ndarray, None]) -> Union[str, None]:
    """
    Convert numpy array to string.

    Use 16-digit numerical precision and no truncation for large arrays.

    Parameters
    ----------
    x: array to convert.

    Returns
    -------
    Array as string.
    """
    if not isinstance(x, np.ndarray):
        return x
    return np.array2string(
        x, threshold=x.size, precision=16, max_line_width=np.inf
    )


def string2ndarray(x: Union[str, float]) -> Union[np.ndarray, float]:
    """
    Convert string to numpy array.

    Parameters
    ----------
    x: array to convert.

    Returns
    -------
    Array as :class:`numpy.ndarray`.
    """
    if not isinstance(x, str):
        return x
    if x.startswith("[["):
        return np.vstack(
            [np.fromstring(xx, sep=" ") for xx in x[2:-2].split("]\n [")]
        )
    else:
        return np.fromstring(x[1:-1], sep=" ")
