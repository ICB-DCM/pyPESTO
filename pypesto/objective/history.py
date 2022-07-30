import abc
import copy
import numbers
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd

from ..C import (
    CHI2,
    EXITFLAG,
    FVAL,
    GRAD,
    HESS,
    HISTORY,
    MESSAGE,
    MESSAGES,
    MODE_FUN,
    MODE_RES,
    N_FVAL,
    N_GRAD,
    N_HESS,
    N_ITERATIONS,
    N_RES,
    N_SRES,
    RES,
    SCHI2,
    SRES,
    SUFFIXES,
    SUFFIXES_CSV,
    SUFFIXES_HDF5,
    TIME,
    TRACE,
    TRACE_SAVE_ITER,
    ModeType,
    X,
)
from ..util import allclose, is_none_or_nan, is_none_or_nan_array, isclose
from .util import (
    chi2_to_fval,
    res_to_chi2,
    schi2_to_grad,
    sres_to_fim,
    sres_to_schi2,
)

ResultDict = Dict[str, Union[float, np.ndarray]]
MaybeArray = Union[np.ndarray, 'np.nan']


class HistoryTypeError(ValueError):
    """Error raised when an unsupported history type is requested."""

    def __init__(self, history_type: str):
        super().__init__(
            f"Unsupported history type: {history_type}, expected {SUFFIXES}"
        )


class CsvHistoryTemplateError(ValueError):
    """Error raised when no template is given for CSV history."""

    def __init__(self, storage_file: str):
        super().__init__(
            "CSV History requires an `{id}` template in the `storage_file`, "
            f"but is {storage_file}"
        )


def trace_wrap(f):
    """
    Wrap around trace getters.

    Transform input `ix` vectors to a valid index list, and reduce for
    integer `ix` the output to a single value.
    """

    def wrapped_f(
        self, ix: Union[Sequence[int], int, None] = None, trim: bool = False
    ) -> Union[Sequence[Union[float, MaybeArray]], Union[float, MaybeArray]]:
        # whether to reduce the output
        reduce = isinstance(ix, numbers.Integral)
        # default: full list
        if ix is None:
            if trim:
                ix = self.get_trimmed_indices()
            else:
                ix = np.arange(0, len(self), dtype=int)
        # turn every input into an index list
        if reduce:
            ix = np.array([ix], dtype=int)
        # obtain the trace
        trace = f(self, ix)
        # reduce the output
        if reduce:
            trace = trace[0]
        return trace

    return wrapped_f


class HistoryOptions(dict):
    """
    Options for what values to record.

    In addition implements a factory pattern to generate history objects.

    Parameters
    ----------
    trace_record:
        Flag indicating whether to record the trace of function calls.
        The trace_record_* flags only become effective if
        trace_record is True.
    trace_record_grad:
        Flag indicating whether to record the gradient in the trace.
    trace_record_hess:
        Flag indicating whether to record the Hessian in the trace.
    trace_record_res:
        Flag indicating whether to record the residual in
        the trace.
    trace_record_sres:
        Flag indicating whether to record the residual sensitivities in
        the trace.
    trace_record_chi2:
        Flag indicating whether to record the chi2 in the trace.
    trace_record_schi2:
        Flag indicating whether to record the chi2 sensitivities in the
        trace.
    trace_save_iter:
        After how many iterations to store the trace.
    storage_file:
        File to save the history to. Can be any of None, a
        "{filename}.csv", or a "{filename}.hdf5" file. Depending on the values,
        the `create_history` method creates the appropriate object.
        Occurrences of "{id}" in the file name are replaced by the `id`
        upon creation of a history, if applicable.
    """

    def __init__(
        self,
        trace_record: bool = False,
        trace_record_grad: bool = True,
        trace_record_hess: bool = True,
        trace_record_res: bool = True,
        trace_record_sres: bool = True,
        trace_record_chi2: bool = True,
        trace_record_schi2: bool = True,
        trace_save_iter: int = 10,
        storage_file: str = None,
    ):
        super().__init__()

        self.trace_record: bool = trace_record
        self.trace_record_grad: bool = trace_record_grad
        self.trace_record_hess: bool = trace_record_hess
        self.trace_record_res: bool = trace_record_res
        self.trace_record_sres: bool = trace_record_sres
        self.trace_record_chi2: bool = trace_record_chi2
        self.trace_record_schi2: bool = trace_record_schi2
        self.trace_save_iter: int = trace_save_iter
        self.storage_file: str = storage_file

        self._sanity_check()

    def __getattr__(self, key):
        """Allow to use keys as attributes."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def _sanity_check(self):
        """Apply basic sanity checks."""
        if self.storage_file is None:
            return

        # extract storage type
        suffix = Path(self.storage_file).suffix[1:]

        # check storage format is valid
        if suffix not in SUFFIXES:
            raise HistoryTypeError(suffix)

        # check csv histories are parametrized
        if suffix in SUFFIXES_CSV and "{id}" not in self.storage_file:
            raise CsvHistoryTemplateError(self.storage_file)

    @staticmethod
    def assert_instance(
        maybe_options: Union['HistoryOptions', Dict],
    ) -> 'HistoryOptions':
        """
        Return a valid options object.

        Parameters
        ----------
        maybe_options: HistoryOptions or dict
        """
        if isinstance(maybe_options, HistoryOptions):
            return maybe_options
        options = HistoryOptions(**maybe_options)
        return options

    def create_history(
        self,
        id: str,
        x_names: Sequence[str],
    ) -> 'HistoryBase':
        """Create a :class:`HistoryBase` object; Factory method.

        Parameters
        ----------
        id:
            Identifier for the history.
        x_names:
            Parameter names.
        """
        # create different history types based on the inputs
        if self.storage_file is None:
            if self.trace_record:
                return MemoryHistory(options=self)
            else:
                return History(options=self)

        # replace id template in storage file
        storage_file = self.storage_file.replace("{id}", id)

        # evaluate type
        suffix = Path(storage_file).suffix[1:]

        # create history type based on storage type
        if suffix in SUFFIXES_CSV:
            return CsvHistory(x_names=x_names, file=storage_file, options=self)
        elif suffix in SUFFIXES_HDF5:
            return Hdf5History(id=id, file=storage_file, options=self)
        else:
            raise HistoryTypeError(suffix)


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


class CsvHistory(History):
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
        If True, history will be initialized from data in the specified file
    """

    def __init__(
        self,
        file: str,
        x_names: Sequence[str] = None,
        options: Union[HistoryOptions, Dict] = None,
        load_from_file: bool = False,
    ):
        super().__init__(options=options)
        self.x_names = x_names
        self._trace: Union[pd.DataFrame, None] = None
        self.file = os.path.abspath(file)

        # create trace file dirs
        if self.file is not None:
            dirname = os.path.dirname(self.file)
            os.makedirs(dirname, exist_ok=True)

        if load_from_file and os.path.exists(self.file):
            trace = pd.read_csv(self.file, header=[0, 1], index_col=0)
            # replace 'nan' in cols with np.NAN
            cols = pd.DataFrame(trace.columns.to_list())
            cols[cols == 'nan'] = np.NaN
            trace.columns = pd.MultiIndex.from_tuples(
                cols.to_records(index=False).tolist()
            )
            for col in trace.columns:
                # transform strings to np.ndarrays
                trace[col] = trace[col].apply(string2ndarray)

            self._trace = trace
            self.x_names = trace[X].columns
            self._update_counts_from_trace()

    def __len__(self) -> int:
        """Define length of history object."""
        return len(self._trace)

    def _update_counts_from_trace(self) -> None:
        self._n_fval = self._trace[(N_FVAL, np.NaN)].max()
        self._n_grad = self._trace[(N_GRAD, np.NaN)].max()
        self._n_hess = self._trace[(N_HESS, np.NaN)].max()
        self._n_res = self._trace[(N_RES, np.NaN)].max()
        self._n_sres = self._trace[(N_SRES, np.NaN)].max()

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

    def finalize(self, message: str = None, exitflag: str = None):
        """See `HistoryBase` docstring."""
        super().finalize()
        self._save_trace(finalize=True)

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
            name=len(self._trace), index=self._trace.columns, dtype='object'
        )

        values = {
            TIME: used_time,
            N_FVAL: self._n_fval,
            N_GRAD: self._n_grad,
            N_HESS: self._n_hess,
            N_RES: self._n_res,
            N_SRES: self._n_sres,
            FVAL: result[FVAL],
            RES: result[RES],
            SRES: result[SRES],
            CHI2: result[CHI2],
            HESS: result[HESS],
        }

        for var, val in values.items():
            row[(var, np.nan)] = val

        for var, val in {
            X: x,
            GRAD: result[GRAD],
            SCHI2: result[SCHI2],
        }.items():
            if var == X or self.options[f'trace_record_{var}']:
                row[var] = val
            else:
                row[(var, np.nan)] = np.nan

        self._trace = pd.concat(
            (self._trace, pd.DataFrame([row])),
        )

        # save trace to file
        self._save_trace()

    def _init_trace(self, x: np.ndarray):
        """Initialize the trace."""
        if self.x_names is None:
            self.x_names = [f'x{i}' for i, _ in enumerate(x)]

        columns: List[Tuple] = [
            (c, np.nan)
            for c in [
                TIME,
                N_FVAL,
                N_GRAD,
                N_HESS,
                N_RES,
                N_SRES,
                FVAL,
                CHI2,
                RES,
                SRES,
                HESS,
            ]
        ]

        for var in [X, GRAD, SCHI2]:
            if var == X or self.options[f'trace_record_{var}']:
                columns.extend([(var, x_name) for x_name in self.x_names])
            else:
                columns.extend([(var,)])

        # TODO: multi-index for res, sres, hess
        self._trace = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(columns), dtype='float64'
        )

        # only non-float64
        trace_dtypes = {
            RES: 'object',
            SRES: 'object',
            HESS: 'object',
            N_FVAL: 'int64',
            N_GRAD: 'int64',
            N_HESS: 'int64',
            N_RES: 'int64',
            N_SRES: 'int64',
        }

        for var, dtype in trace_dtypes.items():
            self._trace[(var, np.nan)] = self._trace[(var, np.nan)].astype(
                dtype
            )

    def _save_trace(self, finalize: bool = False):
        """
        Save to file via pd.DataFrame.to_csv().

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

    @trace_wrap
    def get_x_trace(
        self,
        ix: Union[int, Sequence[int], None] = None,
        trim: bool = False,
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """See `HistoryBase` docstring."""
        return list(self._trace[X].values[ix])

    @trace_wrap
    def get_fval_trace(
        self, ix: Union[int, Sequence[int], None], trim: bool = False
    ) -> Union[Sequence[float], float]:
        """See `HistoryBase` docstring."""
        return list(self._trace[(FVAL, np.nan)].values[ix])

    @trace_wrap
    def get_grad_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return list(self._trace[GRAD].values[ix])

    @trace_wrap
    def get_hess_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return list(self._trace[(HESS, np.nan)].values[ix])

    @trace_wrap
    def get_res_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return list(self._trace[(RES, np.nan)].values[ix])

    @trace_wrap
    def get_sres_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return list(self._trace[(SRES, np.nan)].values[ix])

    @trace_wrap
    def get_chi2_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """See `HistoryBase` docstring."""
        return list(self._trace[(CHI2, np.nan)].values[ix])

    @trace_wrap
    def get_schi2_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return list(self._trace[SCHI2].values[ix])

    @trace_wrap
    def get_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """See `HistoryBase` docstring."""
        return list(self._trace[(TIME, np.nan)].values[ix])


class Hdf5History(History):
    """
    Stores a representation of the history in an HDF5 file.

    Parameters
    ----------
    id:
        Id of the history
    file:
        HDF5 file name.
    options:
        History options.
    """

    def __init__(
        self, id: str, file: str, options: Union[HistoryOptions, Dict] = None
    ):
        super().__init__(options=options)
        self.id: str = id
        self.file: str = file
        self.editable: bool = self._is_editable(file)
        self._generate_hdf5_group()

    def __len__(self) -> int:
        """Define length of history object."""
        with h5py.File(self.file, 'r') as f:
            return f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_ITERATIONS]

    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        """See `History` docstring."""
        if not self.editable:
            raise ValueError(
                f'ID "{self.id}" is already used'
                f' in history file "{self.file}".'
            )
        super().update(x, sensi_orders, mode, result)
        self._update_trace(x, sensi_orders, mode, result)

    def finalize(self, message: str = None, exitflag: str = None) -> None:
        """See `HistoryBase` docstring."""
        super().finalize()

        # add message and exitflag to trace
        with h5py.File(self.file, 'a') as f:
            if f'{HISTORY}/{self.id}/{MESSAGES}/' not in f:
                f.create_group(f'{HISTORY}/{self.id}/{MESSAGES}/')
            grp = f[f'{HISTORY}/{self.id}/{MESSAGES}/']
            if message is not None:
                grp.attrs[MESSAGE] = message
            if exitflag is not None:
                grp.attrs[EXITFLAG] = exitflag

    @staticmethod
    def load(
        id: str, file: str, options: Union[HistoryOptions, Dict] = None
    ) -> 'Hdf5History':
        """Load the History object from memory."""
        history = Hdf5History(id=id, file=file, options=options)
        if options is None:
            history.recover_options(file)
        return history

    def recover_options(self, file: str):
        """Recover options when loading the hdf5 history from memory.

        Done by testing which entries were recorded.
        """
        trace_record = self._has_non_nan_entries(X)
        trace_record_grad = self._has_non_nan_entries(GRAD)
        trace_record_hess = self._has_non_nan_entries(HESS)
        trace_record_res = self._has_non_nan_entries(RES)
        trace_record_sres = self._has_non_nan_entries(SRES)
        trace_record_chi2 = self._has_non_nan_entries(CHI2)
        trace_record_schi2 = self._has_non_nan_entries(SCHI2)

        restored_history_options = HistoryOptions(
            trace_record=trace_record,
            trace_record_grad=trace_record_grad,
            trace_record_hess=trace_record_hess,
            trace_record_res=trace_record_res,
            trace_record_sres=trace_record_sres,
            trace_record_chi2=trace_record_chi2,
            trace_record_schi2=trace_record_schi2,
            trace_save_iter=self.trace_save_iter,
            storage_file=file,
        )

        self.options = restored_history_options

    def _has_non_nan_entries(self, hdf5_group: str) -> bool:
        """Check if there exist non-nan entries stored for a given group."""
        group = self._get_hdf5_entries(hdf5_group, ix=None)

        for entry in group:
            if not (entry is None or np.all(np.isnan(entry))):
                return True

        return False

    # overwrite _update_counts
    def _update_counts(self, sensi_orders: Tuple[int, ...], mode: ModeType):
        """Update the counters in the hdf5."""
        with h5py.File(self.file, 'a') as f:

            if mode == MODE_FUN:
                if 0 in sensi_orders:
                    f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_FVAL] += 1
                if 1 in sensi_orders:
                    f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_GRAD] += 1
                if 2 in sensi_orders:
                    f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_HESS] += 1
            elif mode == MODE_RES:
                if 0 in sensi_orders:
                    f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_RES] += 1
                if 1 in sensi_orders:
                    f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_SRES] += 1

    @property
    def n_fval(self) -> int:
        """See `HistoryBase` docstring."""
        with h5py.File(self.file, 'r') as f:
            return f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_FVAL]

    @property
    def n_grad(self) -> int:
        """See `HistoryBase` docstring."""
        with h5py.File(self.file, 'r') as f:
            return f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_GRAD]

    @property
    def n_hess(self) -> int:
        """See `HistoryBase` docstring."""
        with h5py.File(self.file, 'r') as f:
            return f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_HESS]

    @property
    def n_res(self) -> int:
        """See `HistoryBase` docstring."""
        with h5py.File(self.file, 'r') as f:
            return f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_RES]

    @property
    def n_sres(self) -> int:
        """See `HistoryBase` docstring."""
        with h5py.File(self.file, 'r') as f:
            return f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_SRES]

    @property
    def trace_save_iter(self) -> int:
        """After how many iterations to store the trace."""
        with h5py.File(self.file, 'r') as f:
            return f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[TRACE_SAVE_ITER]

    @property
    def message(self) -> str:
        """Optimizer message in case of finished optimization."""
        with h5py.File(self.file, 'r') as f:
            try:
                return f[f'{HISTORY}/{self.id}/{MESSAGES}/'].attrs[MESSAGE]
            except KeyError:
                return None

    @property
    def exitflag(self) -> str:
        """Optimizer exitflag in case of finished optimization."""
        with h5py.File(self.file, 'r') as f:
            try:
                return f[f'{HISTORY}/{self.id}/{MESSAGES}/'].attrs[EXITFLAG]
            except KeyError:
                return None

    def _update_trace(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        """Update and possibly store the trace."""
        if not self.options.trace_record:
            return

        # calculating function values from residuals
        #  and reduce via requested history options
        result = reduce_result_via_options(
            add_fun_from_res(result), self.options
        )

        used_time = time.time() - self._start_time

        values = {
            X: x,
            FVAL: result[FVAL],
            GRAD: result[GRAD],
            RES: result[RES],
            SRES: result[SRES],
            CHI2: result[CHI2],
            SCHI2: result[SCHI2],
            HESS: result[HESS],
            TIME: used_time,
        }

        with h5py.File(self.file, 'a') as f:
            iteration = f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_ITERATIONS]

            for key in values.keys():
                if values[key] is not None:
                    f[
                        f'{HISTORY}/{self.id}/{TRACE}/{iteration}/{key}'
                    ] = values[key]

            f[f'{HISTORY}/{self.id}/{TRACE}/'].attrs[N_ITERATIONS] += 1

    def _generate_hdf5_group(self, f: h5py.File = None) -> None:
        """Generate group in the hdf5 file, if it does not exist yet."""
        try:
            with h5py.File(self.file, 'a') as f:
                if f'{HISTORY}/{self.id}/{TRACE}/' not in f:
                    grp = f.create_group(f'{HISTORY}/{self.id}/{TRACE}/')
                    grp.attrs[N_ITERATIONS] = 0
                    grp.attrs[N_FVAL] = 0
                    grp.attrs[N_GRAD] = 0
                    grp.attrs[N_HESS] = 0
                    grp.attrs[N_RES] = 0
                    grp.attrs[N_SRES] = 0
                    # TODO Y it makes no sense to save this here
                    #  Also, we do not seem to evaluate this at all
                    grp.attrs[TRACE_SAVE_ITER] = self.options.trace_save_iter
        except OSError:
            pass

    def _get_hdf5_entries(
        self,
        entry_id: str,
        ix: Union[int, Sequence[int], None] = None,
    ) -> Sequence:
        """
        Get entries for field `entry_id` from HDF5 file, for indices `ix`.

        Parameters
        ----------
        entry_id:
            The key whose trace is returned.
        ix:
            Index or list of indices of the iterations that will produce
            the trace.

        Returns
        -------
        The entries ix for the key entry_id.
        """
        if ix is None:
            ix = range(len(self))
        trace_result = []

        with h5py.File(self.file, 'r') as f:
            for iteration in ix:
                try:
                    dataset = f[
                        f'{HISTORY}/{self.id}/{TRACE}/{iteration}/{entry_id}'
                    ]
                    if dataset.shape == ():
                        entry = dataset[()]  # scalar
                    else:
                        entry = np.array(dataset)
                    trace_result.append(entry)
                except KeyError:
                    trace_result.append(None)

        return trace_result

    @trace_wrap
    def get_x_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(X, ix)

    @trace_wrap
    def get_fval_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(FVAL, ix)

    @trace_wrap
    def get_grad_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(GRAD, ix)

    @trace_wrap
    def get_hess_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(HESS, ix)

    @trace_wrap
    def get_res_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(RES, ix)

    @trace_wrap
    def get_sres_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(SRES, ix)

    @trace_wrap
    def get_chi2_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(CHI2, ix)

    @trace_wrap
    def get_schi2_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[MaybeArray], MaybeArray]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(SCHI2, ix)

    @trace_wrap
    def get_time_trace(
        self, ix: Union[int, Sequence[int], None] = None, trim: bool = False
    ) -> Union[Sequence[float], float]:
        """See `HistoryBase` docstring."""
        return self._get_hdf5_entries(TIME, ix)

    def _is_editable(self, file: str) -> bool:
        """
        Check whether the id is already existent in the file.

        Parameters
        ----------
        file:
            HDF5 file name.

        Returns
        -------
        editable:
            Boolean, whether this hdf5 file should be editable.
            Returns true if the file or the id entry does not exist yet.
        """
        try:
            with h5py.File(file, 'r') as f:
                # editable if the id entry does not exist
                return 'history' not in f.keys() or self.id not in f['history']
        except OSError:
            # editable if the file does not exist
            return True


class OptimizerHistory:
    """
    Optimizer objective call history.

    Container around a History object, additionally keeping track of optimal
    values.

    Attributes
    ----------
    fval0, fval_min:
        Initial and best function value found.
    chi20, chi2_min:
        Initial and best chi2 value found.
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
        function based on the provided history.
    """

    # optimal point values
    MIN_KEYS = (X, *HistoryBase.RESULT_KEYS)

    def __init__(
        self,
        history: History,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        generate_from_history: bool = False,
    ) -> None:
        self.history: History = history

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
            self._compute_vals_from_trace()

    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int],
        mode: ModeType,
        result: ResultDict,
    ) -> None:
        """Update history and best found value."""
        result = add_fun_from_res(result)
        self._update_vals(x, result)
        self.history.update(x, sensi_orders, mode, result)

    def finalize(self, message: str = None, exitflag: int = None):
        """
        Finalize history.

        Parameters
        ----------
        message:
            Optimizer message to be saved.
        exitflag:
            Optimizer exitflag to be saved.
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
            # update everything
            for key in self.MIN_KEYS:
                setattr(self, key + '_min', result[key])

        # check if history has same point
        if isclose(fval, self.fval_min) and allclose(result[X], self.x_min):
            # update only missing entries
            #  (e.g. grad and hess may be recorded but not in history)
            for key in self.MIN_KEYS:
                if result[key] is not None:
                    # if getattr(self, f'{key}_min') is None:
                    setattr(self, f'{key}_min', result[key])

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
                setattr(self, f'{key}_min', result.get(key))
            self.x_min = x
            return

        # Sometimes sensitivities are evaluated on subsequent calls. We can
        # identify this situation by checking that x hasn't changed.
        if self.x_min is not None and np.array_equal(self.x_min, x):
            for key in (GRAD, HESS, SRES):
                val_min = getattr(self, f'{key}_min', None)
                val = result.get(key)
                if is_none_or_nan_array(val_min) and not is_none_or_nan_array(
                    val
                ):
                    setattr(self, f'{key}_min', result.get(key))

    def _compute_vals_from_trace(self) -> None:
        """Set initial and best function value from trace."""
        if not len(self.history):
            # nothing to be computed from empty history
            return

        # some optimizers may evaluate hess+grad first to compute trust region
        # etc
        for it in range(len(self.history)):
            fval = self.history.get_fval_trace(it)
            x = self.history.get_x_trace(it)
            if not is_none_or_nan(fval) and allclose(x, self.x0):
                self.fval0 = float(fval)
                break

        # find best fval
        result = self._get_optimal_point_from_history()

        # assign values
        for key in OptimizerHistory.MIN_KEYS:
            setattr(self, f'{key}_min', result[key])

    def _admissible(self, x: np.ndarray) -> bool:
        """Check whether point `x` is admissible (i.e. within bounds).

        Parameters
        ----------
        x: A single parameter vector.

        Returns
        -------
        admissible: Whether the point fulfills the problem requirements.
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
            val = getattr(self.history, f'get_{var}_trace')(ix_min)
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
                val = getattr(self.history, f'get_{var}_trace')(ix)
                if not is_none_or_nan_array(val):
                    result[var] = val
                    # successfuly found
                    break

        # fill remaining keys with None
        for key in OptimizerHistory.MIN_KEYS:
            if key not in result:
                result[key] = None

        return result


def ndarray2string_full(x: Union[np.ndarray, None]) -> Union[str, None]:
    """
    Convert numpy array to string.

    Use 16-digit numerical precision and no truncation for large arrays.

    Parameters
    ----------
    x: array to convert.

    Returns
    -------
    x: array as string.
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
    x: array as np.ndarray.
    """
    if not isinstance(x, str):
        return x
    if x.startswith('[['):
        return np.vstack(
            [np.fromstring(xx, sep=' ') for xx in x[2:-2].split(']\n [')]
        )
    else:
        return np.fromstring(x[1:-1], sep=' ')


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
