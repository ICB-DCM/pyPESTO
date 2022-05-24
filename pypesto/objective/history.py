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
    FVAL,
    GRAD,
    HESS,
    MODE_FUN,
    MODE_RES,
    N_FVAL,
    N_GRAD,
    N_HESS,
    N_RES,
    N_SRES,
    RES,
    SCHI2,
    SRES,
    TIME,
    X,
)
from .util import (
    res_to_chi2,
    res_to_fval,
    schi2_to_grad,
    sres_to_fim,
    sres_to_schi2,
)

ResultDict = Dict[str, Union[float, np.ndarray]]
MaybeArray = Union[np.ndarray, 'np.nan']


def trace_wrap(f):
    """
    Wrap around trace getters.

    Transform input `ix` vectors to a valid index list, and reduces for
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
    Options for the objective that are used in optimization.

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
        type_ = Path(self.storage_file).suffix

        # check storage format is valid
        if type_ not in [".csv", ".hdf5", ".h5"]:
            raise ValueError(
                "Only history storage to '.csv' and '.hdf5' is supported, got "
                f"{type_}",
            )

        # check csv histories are parametrized
        if type_ == ".csv" and "{id}" not in self.storage_file:
            raise ValueError(
                "For csv history, the `storage_file` must contain an `{id}` "
                "template"
            )

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
    ) -> 'History':
        """Create a :class:`History` object; Factory method.

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

        storage_file = self.storage_file.replace("{id}", id)

        _, type_ = os.path.splitext(storage_file)

        if type_ == '.csv':
            return CsvHistory(x_names=x_names, file=storage_file, options=self)
        elif type_ in ['.hdf5', '.h5']:
            return Hdf5History(id=id, file=storage_file, options=self)
        else:
            raise ValueError(
                "Only history storage to '.csv' and '.hdf5' is supported, got "
                f"{type_}",
            )


class HistoryBase(abc.ABC):
    """Abstract base class for history objects.

    Can be used as a dummy history, but does not implement any history
    functionality.
    """

    def __len__(self):
        """Define length by number of stored entries in the history."""
        raise NotImplementedError()

    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: str,
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
    ):
        """
        Finalize history. Called after a run.

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

    def get_trimmed_indices(self):
        """Get indices for a monotonically decreasing history."""
        fval_trace = self.get_fval_trace()
        return np.where(fval_trace <= np.fmin.accumulate(fval_trace))[0]


class History(HistoryBase):
    """
    Track number of function evaluations only, no trace.

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
        mode: str,
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
        res = result.get(RES, None)
        if res is not None and FVAL not in result:
            # no option trace_record_fval
            result[FVAL] = res_to_fval(res)
        self._update_counts(sensi_orders, mode)

    def finalize(self, message: str = None, exitflag: str = None):
        """See `HistoryBase` docstring."""
        pass

    def _update_counts(
        self,
        sensi_orders: Tuple[int, ...],
        mode: str,
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

    Track number of function evaluations and keeps an in-memory
    trace of function evaluations.

    Parameters
    ----------
    options:
        History options.
    """

    def __init__(self, options: Union[HistoryOptions, Dict] = None):
        super().__init__(options=options)
        self._trace_keys = {X, FVAL, GRAD, HESS, RES, SRES, CHI2, SCHI2, TIME}
        self._trace: Dict[str, Any] = {key: [] for key in self._trace_keys}

    def __len__(self):
        """Define length of history object."""
        return len(self._trace[TIME])

    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: str,
        result: ResultDict,
    ) -> None:
        """See `History` docstring."""
        super().update(x, sensi_orders, mode, result)
        self._update_trace(x, mode, result)

    def _update_trace(self, x, mode, result):
        """Update internal trace representation."""
        ret = extract_values(mode, result, self.options)
        for key in self._trace_keys - {X, TIME}:
            self._trace[key].append(ret[key])
        used_time = time.time() - self._start_time
        self._trace[X].append(x)
        self._trace[TIME].append(used_time)

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

    def __len__(self):
        """Define length of history object."""
        return len(self._trace)

    def _update_counts_from_trace(self):
        self._n_fval = self._trace[('n_fval', np.NaN)].max()
        self._n_grad = self._trace[('n_grad', np.NaN)].max()
        self._n_hess = self._trace[('n_hess', np.NaN)].max()
        self._n_res = self._trace[('n_res', np.NaN)].max()
        self._n_sres = self._trace[('n_sres', np.NaN)].max()

    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: str,
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
        mode: str,
        result: ResultDict,
    ):
        """Update and possibly store the trace."""
        if not self.options.trace_record:
            return

        # init trace
        if self._trace is None:
            self._init_trace(x)

        # extract function values
        ret = extract_values(mode, result, self.options)

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
            FVAL: ret[FVAL],
            RES: ret[RES],
            SRES: ret[SRES],
            CHI2: ret[CHI2],
            HESS: ret[HESS],
        }

        for var, val in values.items():
            row[(var, float('nan'))] = val

        for var, val in {X: x, GRAD: ret[GRAD], SCHI2: ret[SCHI2]}.items():
            if var == X or self.options[f'trace_record_{var}']:
                row[var] = val
            else:
                row[(var, float('nan'))] = np.NaN

        self._trace = self._trace.append(row)

        # save trace to file
        self._save_trace()

    def _init_trace(self, x: np.ndarray):
        """Initialize the trace."""
        if self.x_names is None:
            self.x_names = [f'x{i}' for i, _ in enumerate(x)]

        columns: List[Tuple] = [
            (c, float('nan'))
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
            if var == 'x' or self.options[f'trace_record_{var}']:
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
            self._trace[(var, np.NaN)] = self._trace[(var, np.NaN)].astype(
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
            for field in [('hess', np.NaN), ('res', np.NaN), ('sres', np.NaN)]:
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
        self.id = id
        self.file, self.editable = self._check_file_id(file)
        self._generate_hdf5_group()

    def __len__(self):
        """Define length of history object."""
        with h5py.File(self.file, 'r') as f:
            return f[f'history/{self.id}/trace/'].attrs['n_iterations']

    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: str,
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

    def get_history_directory(self):
        """Return filepath."""
        return self.file

    def finalize(self, message: str = None, exitflag: str = None):
        """See `HistoryBase` docstring."""
        super().finalize()
        with h5py.File(self.file, 'a') as f:
            if f'history/{self.id}/messages/' not in f:
                f.create_group(f'history/{self.id}/messages/')
            grp = f[f'history/{self.id}/messages/']
            if message is not None:
                grp.attrs['message'] = message
            if exitflag is not None:
                grp.attrs['exitflag'] = exitflag

    @staticmethod
    def load(id: str, file: str):
        """Load the History object from memory."""
        loaded_h5history = Hdf5History(id, file)
        loaded_h5history.recover_options(file)
        return loaded_h5history

    def recover_options(self, file: str):
        """Recover options when loading the hdf5 history from memory.

        Done by testing which entries were recorded.
        """
        trace_record = self._check_for_not_nan_entries(X)
        trace_record_grad = self._check_for_not_nan_entries(GRAD)
        trace_record_hess = self._check_for_not_nan_entries(HESS)
        trace_record_res = self._check_for_not_nan_entries(RES)
        trace_record_sres = self._check_for_not_nan_entries(SRES)
        trace_record_chi2 = self._check_for_not_nan_entries(CHI2)
        trace_record_schi2 = self._check_for_not_nan_entries(SCHI2)
        storage_file = file

        restored_history_options = HistoryOptions(
            trace_record=trace_record,
            trace_record_grad=trace_record_grad,
            trace_record_hess=trace_record_hess,
            trace_record_res=trace_record_res,
            trace_record_sres=trace_record_sres,
            trace_record_chi2=trace_record_chi2,
            trace_record_schi2=trace_record_schi2,
            trace_save_iter=self.trace_save_iter,
            storage_file=storage_file,
        )

        self.options = restored_history_options

    def _check_for_not_nan_entries(self, hdf5_group: str) -> bool:
        """Check if there exist not-nan entries stored for a given group."""
        group = self._get_hdf5_entries(hdf5_group, ix=None)

        for entry in group:
            if not (entry is None or np.all(np.isnan(entry))):
                return True

        return False

    # overwrite _update_counts
    def _update_counts(self, sensi_orders: Tuple[int, ...], mode: str):
        """Update the counters in the hdf5."""
        with h5py.File(self.file, 'a') as f:

            if mode == MODE_FUN:
                if 0 in sensi_orders:
                    f[f'history/{self.id}/trace/'].attrs['n_fval'] += 1
                if 1 in sensi_orders:
                    f[f'history/{self.id}/trace/'].attrs['n_grad'] += 1
                if 2 in sensi_orders:
                    f[f'history/{self.id}/trace/'].attrs['n_hess'] += 1
            elif mode == MODE_RES:
                if 0 in sensi_orders:
                    f[f'history/{self.id}/trace/'].attrs['n_res'] += 1
                if 1 in sensi_orders:
                    f[f'history/{self.id}/trace/'].attrs['n_sres'] += 1

    @property
    def n_fval(self) -> int:
        """See `HistoryBase` docstring."""
        with h5py.File(self.file, 'r') as f:
            return f[f'history/{self.id}/trace/'].attrs['n_fval']

    @property
    def n_grad(self) -> int:
        """See `HistoryBase` docstring."""
        with h5py.File(self.file, 'r') as f:
            return f[f'history/{self.id}/trace/'].attrs['n_grad']

    @property
    def n_hess(self) -> int:
        """See `HistoryBase` docstring."""
        with h5py.File(self.file, 'r') as f:
            return f[f'history/{self.id}/trace/'].attrs['n_hess']

    @property
    def n_res(self) -> int:
        """See `HistoryBase` docstring."""
        with h5py.File(self.file, 'r') as f:
            return f[f'history/{self.id}/trace/'].attrs['n_res']

    @property
    def n_sres(self) -> int:
        """See `HistoryBase` docstring."""
        with h5py.File(self.file, 'r') as f:
            return f[f'history/{self.id}/trace/'].attrs['n_sres']

    @property
    def trace_save_iter(self):
        """After how many iterations to store the trace."""
        with h5py.File(self.file, 'r') as f:
            return f[f'history/{self.id}/trace/'].attrs['trace_save_iter']

    @property
    def message(self):
        """Optimizer message in case of finished optimization."""
        with h5py.File(self.file, 'r') as f:
            try:
                return f[f'history/{self.id}/messages/'].attrs['message']
            except KeyError:
                return None

    @property
    def exitflag(self):
        """Optimizer exitflag in case of finished optimization."""
        with h5py.File(self.file, 'r') as f:
            try:
                return f[f'history/{self.id}/messages/'].attrs['exitflag']
            except KeyError:
                return None

    def _update_trace(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int],
        mode: str,
        result: ResultDict,
    ):
        """Update and possibly store the trace."""
        if not self.options.trace_record:
            return

        # extract function values
        ret = extract_values(mode, result, self.options)

        used_time = time.time() - self._start_time

        values = {
            TIME: used_time,
            X: x,
            FVAL: ret[FVAL],
            GRAD: ret[GRAD],
            RES: ret[RES],
            SRES: ret[SRES],
            CHI2: ret[CHI2],
            SCHI2: ret[SCHI2],
            HESS: ret[HESS],
        }

        with h5py.File(self.file, 'a') as f:

            iteration = f[f'history/{self.id}/trace/'].attrs['n_iterations']

            for key in values.keys():
                if values[key] is not None:
                    f[
                        f'history/{self.id}/trace/' f'{str(iteration)}/{key}'
                    ] = values[key]

            f[f'history/{self.id}/trace/'].attrs['n_iterations'] += 1

    def _generate_hdf5_group(self, f: h5py.File = None):
        """Generate the group in the hdf5 file, if it does not exist yet."""
        try:
            with h5py.File(self.file, 'a') as f:
                if f'history/{self.id}/trace/' not in f:
                    grp = f.create_group(f'history/{self.id}/trace/')
                    grp.attrs['n_iterations'] = 0
                    grp.attrs['n_fval'] = 0
                    grp.attrs['n_grad'] = 0
                    grp.attrs['n_hess'] = 0
                    grp.attrs['n_res'] = 0
                    grp.attrs['n_sres'] = 0
                    grp.attrs['trace_save_iter'] = self.options.trace_save_iter
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
                        f'history/{self.id}/trace/{str(iteration)}/{entry_id}'
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

    def _check_file_id(self, file: str):
        """
        Check, whether the id is already existent in the file.

        Parameters
        ----------
        file:
            HDF5 file name.

        Returns
        -------
        file:
            HDF5 file name.
        editable:
            Boolean, whether this hdf5 file should be editable. Returns
            false if the history is a loaded one to prevent overwriting.

        """
        try:
            with h5py.File(file, 'r') as f:
                return file, (
                    'history' not in f.keys() or self.id not in f['history']
                )
        except OSError:  # if the file is non-existent, return editable = True
            return file, True


class OptimizerHistory:
    """
    Objective call history.

    Container around a History object, which keeps track of optimal values.

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
        mode: str,
        result: ResultDict,
    ) -> None:
        """Update history and best found value."""
        self.history.update(x, sensi_orders, mode, result)
        self._update_vals(x, result)

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

    def _update_vals(self, x: np.ndarray, result: ResultDict):
        """Update initial and best function values."""
        # update initial point
        if np.allclose(x, self.x0):
            if self.fval0 is None:
                self.fval0 = result.get(FVAL, None)
            self.x0 = x

        # don't update optimal point if point is not admissible
        if not self._admissible(x):
            return

        # update best point
        fval = result.get(FVAL, None)
        grad = result.get(GRAD, None)
        hess = result.get(HESS, None)
        res = result.get(RES, None)
        sres = result.get(SRES, None)

        if fval is not None and fval < self.fval_min:
            self.fval_min = fval
            self.x_min = x
            self.grad_min = grad
            self.hess_min = hess
            self.res_min = res
            self.sres_min = sres

        # sometimes sensitivities are evaluated on subsequent calls. We can
        # identify this situation by checking that x hasn't changed
        if self.x_min is not None and np.allclose(self.x_min, x):
            if self.grad_min is None and grad is not None:
                self.grad_min = grad
            if self.hess_min is None and hess is not None:
                self.hess_min = hess
            if self.res_min is None and res is not None:
                self.res_min = res
            if self.sres_min is None and sres is not None:
                self.sres_min = sres

    def _compute_vals_from_trace(self):
        """Set initial and best function value from trace (at start)."""
        if not len(self.history):
            # nothing to be computed from empty history
            return

        # some optimizers may evaluate hess+grad first to compute trust region
        # etc
        max_init_iter = 3
        for it in range(min(len(self.history), max_init_iter)):
            candidate = self.history.get_fval_trace(it)
            if not np.isnan(candidate) and np.allclose(
                self.history.get_x_trace(it), self.x0
            ):
                self.fval0 = float(candidate)
                break

        # get indices of admissible trace entries
        # shape (n_sample, n_x)
        xs = np.asarray(self.history.get_x_trace())
        ixs_admit = [ix for ix, x in enumerate(xs) if self._admissible(x)]

        # we prioritize fval over chi2 as fval is written whenever possible
        ix_min = np.nanargmin(self.history.get_fval_trace(ixs_admit))
        # np.argmin returns ndarray when multiple minimal values are found, we
        # generally want the first occurrence
        if isinstance(ix_min, np.ndarray):
            ix_min = ix_min[0]
        # select index in original array
        ix_min = ixs_admit[ix_min]

        for var in ['fval', 'chi2', 'x']:
            self.extract_from_history(var, ix_min)
            if var == 'fval':
                self.fval_min = float(self.fval_min)

        for var in ['res', 'grad', 'sres', 'hess']:
            if not getattr(self.history.options, f'trace_record_{var}'):
                continue  # var not saved in history
            # first try index of optimal function value
            if self.extract_from_history(var, ix_min):
                continue
            # gradients may be evaluated at different indices, therefore
            #  iterate over all and check whether any has the same parameter
            #  and the desired field filled
            # for res we do the same because otherwise randomly None
            #  (TODO investigate why, but ok this way)
            for ix in reversed(range(len(self.history))):
                if not np.allclose(self.x_min, self.history.get_x_trace(ix)):
                    continue
                if self.extract_from_history(var, ix):
                    # successfully assigned
                    break

    def extract_from_history(self, var: str, ix: int) -> bool:
        """Get value of `var` at iteration `ix` and assign to `{var}_min`.

        Parameters
        ----------
        var: Variable to extract, e.g. 'grad', 'x'.
        ix: Trace index.

        Returns
        -------
        successful:
            Whether extraction and assignment worked. False in particular if
            the history value is nan.
        """
        val = getattr(self.history, f'get_{var}_trace')(ix)
        if not np.all(np.isnan(val)):
            setattr(self, f'{var}_min', val)
            return True
        return False

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


def ndarray2string_full(x: Union[np.ndarray, None]) -> Union[str, None]:
    """
    Convert numpy arrays to string.

    Use 16 digit numerical precision and no truncation for large arrays

    Parameters
    ----------
    x:
        array to convert

    Returns
    -------
    x:
        array as string
    """
    if not isinstance(x, np.ndarray):
        return x
    return np.array2string(
        x, threshold=x.size, precision=16, max_line_width=np.inf
    )


def string2ndarray(x: Union[str, float]) -> Union[np.ndarray, float]:
    """
    Convert string to numpy arrays.

    Parameters
    ----------
    x:
        array to convert

    Returns
    -------
    x:
        array as np.ndarray
    """
    if not isinstance(x, str):
        return x
    if x.startswith('[['):
        return np.vstack(
            [np.fromstring(xx, sep=' ') for xx in x[2:-2].split(']\n [')]
        )
    else:
        return np.fromstring(x[1:-1], sep=' ')


def extract_values(
    mode: str, result: ResultDict, options: HistoryOptions
) -> Dict:
    """Extract values to record from result."""
    ret = {}
    ret_vars = [FVAL, GRAD, HESS, RES, SRES, CHI2, SCHI2]
    for var in ret_vars:
        if options.get(f'trace_record_{var}', True) and var in result:
            ret[var] = result[var]

    # write values that weren't set yet with alternative methods
    if mode == MODE_RES:
        res_result = result.get(RES, None)
        sres_result = result.get(SRES, None)
        chi2 = res_to_chi2(res_result)
        schi2 = sres_to_schi2(res_result, sres_result)
        fim = sres_to_fim(sres_result)
        alt_values = {CHI2: chi2, SCHI2: schi2, HESS: fim}
        if schi2 is not None:
            alt_values[GRAD] = schi2_to_grad(schi2)

        # filter according to options
        alt_values = {
            key: val
            for key, val in alt_values.items()
            if options.get(f'trace_record_{key}', True)
        }
        for var, val in alt_values.items():
            if val is not None:
                ret[var] = ret.get(var, val)

    # set everything missing to NaN
    for var in ret_vars:
        if var not in ret:
            ret[var] = np.NaN

    return ret
