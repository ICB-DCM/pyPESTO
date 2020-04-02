import numpy as np
import pandas as pd
import copy
import time
import os
import abc
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence, Union

from .constants import (
    MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, CHI2, SCHI2, TIME,
    N_FVAL, N_GRAD, N_HESS, N_RES, N_SRES, X)
from .util import res_to_chi2, sres_to_schi2, sres_to_fim

ResultType = Dict[str, Union[float, np.ndarray]]


class HistoryOptions(dict):
    """
    Options for the objective that are used in optimization, profiles
    and sampling.

    In addition implements a factory pattern to generate history objects.

    Parameters
    ----------
    trace_record:
        Flag indicating whether to record the trace of function calls.
        The trace_record_* flags only become effective if
        trace_record is True.
        Default: False.
    trace_record_grad:
        Flag indicating whether to record the gradient in the trace.
        Default: True.
    trace_record_hess:
        Flag indicating whether to record the Hessian in the trace.
        Default: False.
    trace_record_res:
        Flag indicating whether to record the residual in
        the trace.
        Default: False.
    trace_record_sres:
        Flag indicating whether to record the residual sensitivities in
        the trace.
        Default: False.
    trace_record_chi2:
        Flag indicating whether to record the chi2 in the trace.
        Default: True.
    trace_record_schi2:
        Flag indicating whether to record the chi2 sensitivities in the
        trace.
        Default: True.
    trace_save_iter:
        After how many iterations to store the trace.
    storage_file:
        File to save the history to. Can be any of None, a
        "{filename}.csv", or a "{filename}.hdf5" file. Depending on the values,
        the `create_history` method creates the appropriate object.
        Occurrences of "{id}" in the file name are replaced by the `id`
        upon creation of a history, if applicable.
    """

    def __init__(self,
                 trace_record: bool = False,
                 trace_record_grad: bool = True,
                 trace_record_hess: bool = True,
                 trace_record_res: bool = True,
                 trace_record_sres: bool = True,
                 trace_record_chi2: bool = True,
                 trace_record_schi2: bool = True,
                 trace_save_iter: int = 10,
                 storage_file: str = None):
        super().__init__()
        self.trace_record = trace_record
        self.trace_record_grad = trace_record_grad
        self.trace_record_hess = trace_record_hess
        self.trace_record_res = trace_record_res
        self.trace_record_sres = trace_record_sres
        self.trace_record_chi2 = trace_record_chi2
        self.trace_record_schi2 = trace_record_schi2
        self.trace_save_iter = trace_save_iter
        self.storage_file = storage_file

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def assert_instance(
            maybe_options: Union['HistoryOptions', Dict]
    ) -> 'HistoryOptions':
        """
        Returns a valid options object.

        Parameters
        ----------
        maybe_options: HistoryOptions or dict
        """
        if isinstance(maybe_options, HistoryOptions):
            return maybe_options
        options = HistoryOptions(**maybe_options)
        return options

    def create_history(
            self, id: str, x_names: Iterable[str]
    ) -> 'History':
        """Factory method creating a :class:`History` object.

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

        _, type = os.path.splitext(storage_file)

        if type == '.csv':
            return CsvHistory(
                x_names=x_names,
                file=storage_file, options=self)
        elif type == '.hdf5':
            return Hdf5History(id=id, file=storage_file, options=self)
        else:
            raise ValueError(
                "Currently only history storage to '.csv' and '.hdf5'"
                "is supported")


class HistoryBase(abc.ABC):
    """Abstract base class for history objects.

    Can be used as a dummy history, but does not implement any history
    functionality.
    """

    def update(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...],
            mode: str,
            result: ResultType
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

    def finalize(self):
        """Finalize history. Called after a run."""

    @property
    def n_fval(self) -> int:
        """Number of function evaluations."""
        raise NotImplementedError()

    @property
    def n_grad(self) -> int:
        """Number of gradient evaluations."""
        raise NotImplementedError()

    @property
    def n_hess(self) -> int:
        """Number of Hessian evaluations."""
        raise NotImplementedError()

    @property
    def n_res(self) -> int:
        """Number of residual evaluations."""
        raise NotImplementedError()

    @property
    def n_sres(self) -> int:
        """Number or residual sensitivity evaluations."""
        raise NotImplementedError()

    @property
    def start_time(self) -> float:
        """Start time."""
        raise NotImplementedError()

    def get_x_trace(self) -> Sequence[np.ndarray]:
        """Parameter trace."""
        raise NotImplementedError()

    def get_fval_trace(self) -> Sequence[float]:
        """Function value trace."""
        raise NotImplementedError()

    def get_grad_trace(self) -> Sequence[np.ndarray]:
        """Gradient trace."""
        raise NotImplementedError()

    def get_hess_trace(self) -> Sequence[np.ndarray]:
        """Hessian trace."""
        raise NotImplementedError()

    def get_res_trace(self) -> Sequence[np.ndarray]:
        """Residual trace."""
        raise NotImplementedError()

    def get_sres_trace(self) -> Sequence[np.ndarray]:
        """Residual sensitivity trace."""
        raise NotImplementedError()

    def get_chi2_trace(self) -> Sequence[np.ndarray]:
        """Chi2 value trace."""
        raise NotImplementedError()

    def get_schi2_trace(self, t: Optional[int] = None) -> Sequence[np.ndarray]:
        """Chi2 value sensitivity trace."""
        raise NotImplementedError()

    def get_time_trace(self, t: Optional[int] = None) -> Sequence[np.ndarray]:
        """Execution time trace."""
        raise NotImplementedError()


class History(HistoryBase):
    """Tracks numbers of function evaluations only, no trace.

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
            result: ResultType
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

    def finalize(self):
        pass

    def _update_counts(self,
                       sensi_orders: Tuple[int, ...],
                       mode: str):
        """
        Update the counters.
        """
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
        return self._n_fval

    @property
    def n_grad(self) -> int:
        return self._n_grad

    @property
    def n_hess(self) -> int:
        return self._n_hess

    @property
    def n_res(self) -> int:
        return self._n_res

    @property
    def n_sres(self) -> int:
        return self._n_sres

    @property
    def start_time(self) -> float:
        return self._start_time


class MemoryHistory(History):
    """Tracks numbers of function evaluations and keeps an in-memory
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

    def update(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...],
            mode: str,
            result: ResultType
    ) -> None:
        super().update(x, sensi_orders, mode, result)
        self._update_trace(x, sensi_orders, mode, result)

    def _update_trace(self, x, sensi_orders, mode, result):
        """Update internal trace representation."""
        ret = extract_values(sensi_orders, mode, result, self.options)
        for key in self._trace_keys - {X, TIME}:
            self._trace[key].append(ret[key])
        used_time = time.time() - self._start_time
        self._trace[X].append(x)
        self._trace[TIME].append(used_time)

    def get_x_trace(self) -> Sequence[np.ndarray]:
        return self._trace[X]

    def get_fval_trace(self) -> Sequence[float]:
        return self._trace[FVAL]

    def get_grad_trace(self) -> Sequence[np.ndarray]:
        return self._trace[GRAD]

    def get_hess_trace(self) -> Sequence[np.ndarray]:
        return self._trace[HESS]

    def get_res_trace(self) -> Sequence[np.ndarray]:
        return self._trace[RES]

    def get_sres_trace(self) -> Sequence[np.ndarray]:
        return self._trace[SRES]

    def get_chi2_trace(self) -> Sequence[np.ndarray]:
        return self._trace[CHI2]

    def get_schi2_trace(self, t: Optional[int] = None) -> Sequence[np.ndarray]:
        return self._trace[SCHI2]

    def get_time_trace(self, t: Optional[int] = None) -> Sequence[np.ndarray]:
        return self._trace[TIME]


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
    """

    def __init__(self,
                 file: str,
                 x_names: Iterable[str] = None,
                 options: Union[HistoryOptions, Dict] = None):
        super().__init__(options=options)
        self.x_names = x_names
        self._trace: Union[pd.DataFrame, None] = None
        self.file = os.path.abspath(file)

        # create trace file dirs
        if self.file is not None:
            dirname = os.path.dirname(self.file)
            os.makedirs(dirname, exist_ok=True)

    def update(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...],
            mode: str,
            result: ResultType
    ) -> None:
        super().update(x, sensi_orders, mode, result)
        self._update_trace(x, sensi_orders, mode, result)

    def finalize(self):
        """Finalize history. Called after a run."""
        super().finalize()
        self._save_trace(finalize=True)

    def _update_trace(self,
                      x: np.ndarray,
                      sensi_orders: Tuple[int],
                      mode: str,
                      result: ResultType):
        """
        Update and possibly store the trace.
        """
        if not self.options.trace_record:
            return

        # init trace
        if self._trace is None:
            self._init_trace(x)

        # extract function values
        ret = extract_values(sensi_orders, mode, result, self.options)

        used_time = time.time() - self._start_time

        # create table row
        row = pd.Series(name=len(self._trace),
                        index=self._trace.columns,
                        dtype='object')

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
                row[var] = val if val is not None else np.NaN
            else:
                row[(var, float('nan'))] = None

        self._trace = self._trace.append(row)

        # save trace to file
        self._save_trace()

    def _init_trace(self, x: np.ndarray):
        """
        Initialize the trace.
        """
        if self.x_names is None:
            self.x_names = [f'x{i}' for i, _ in enumerate(x)]

        columns: List[Tuple] = [
            (c, float('nan')) for c in [
                TIME, N_FVAL, N_GRAD, N_HESS, N_RES, N_SRES,
                FVAL, CHI2, RES, SRES, HESS,
            ]
        ]

        for var in [X, GRAD, SCHI2]:
            if var == 'x' or self.options[f'trace_record_{var}']:
                columns.extend([
                    (var, x_name)
                    for x_name in self.x_names
                ])
            else:
                columns.extend([(var,)])

        # TODO: multi-index for res, sres, hess

        self._trace = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns),
                                   dtype='float64')

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
            self._trace[(var, float('nan'))] = \
                self._trace[(var, float('nan'))].astype(dtype)

    def _save_trace(self, finalize: bool = False):
        """
        Save to file via pd.DataFrame.to_csv() if `self.storage_file` is
        not None and other conditions apply.

        .. note::
            Format might be revised when storage is implemented.
        """
        if self.file is None:
            return

        if finalize \
                or (len(self._trace) > 0 and len(self._trace) %
                    self.options.trace_save_iter == 0):
            # save
            trace_copy = copy.deepcopy(self._trace)
            for field in [('hess', np.NaN), ('res', np.NaN), ('sres', np.NaN)]:
                trace_copy[field] = trace_copy[field].apply(
                    ndarray2string_full
                )
            trace_copy.to_csv(self.file)

    def get_fval_trace(self) -> pd.Series:
        # TODO implement the other methods
        return self._trace[FVAL]


class Hdf5History(History):
    """Stores a representation of the history in an HDF5 file.

    Parameters
    ----------
    id:
        Id of the history
    file:
        HDF5 file name.
    options:
        History options.
    """

    def __init__(self,
                 id: str,
                 file: str,
                 options: Union[HistoryOptions, Dict] = None):
        super().__init__(options=options)
        self.id = id
        self.file = file

    def update(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...],
            mode: str,
            result: ResultType
    ) -> None:
        # TODO implement
        raise NotImplementedError()

    def finalize(self):
        # TODO implement
        raise NotImplementedError()

    def get_x_trace(self) -> Sequence[np.ndarray]:
        # TODO implement
        raise NotImplementedError()

    def get_fval_trace(self) -> Sequence[float]:
        # TODO implement
        raise NotImplementedError()

    def get_grad_trace(self) -> Sequence[np.ndarray]:
        # TODO implement
        raise NotImplementedError()

    def get_hess_trace(self) -> Sequence[np.ndarray]:
        # TODO implement
        raise NotImplementedError()

    def get_res_trace(self) -> Sequence[np.ndarray]:
        # TODO implement
        raise NotImplementedError()

    def get_sres_trace(self) -> Sequence[np.ndarray]:
        # TODO implement
        raise NotImplementedError()

    def get_chi2_trace(self) -> Sequence[np.ndarray]:
        # TODO implement
        raise NotImplementedError()

    def get_schi2_trace(self, t: Optional[int] = None) -> Sequence[np.ndarray]:
        # TODO implement
        raise NotImplementedError()

    def get_time_trace(self, t: Optional[int] = None) -> Sequence[np.ndarray]:
        # TODO implement
        raise NotImplementedError()


class OptimizerHistory(HistoryBase):
    """
    Objective call history. Also handles saving of intermediate results.

    Attributes
    ----------
    fval0, fval_min:
        Initial and best function value found.
    x0, x_min:
        Initial and best parameters found.
    """

    def __init__(self,
                 history: History,
                 x0: np.ndarray) -> None:
        self.history: History = history

        # initial point
        self.fval0: Union[float, None] = None
        self.x0: np.ndarray = x0

        # minimum point
        self.fval_min: float = np.inf
        self.x_min: Union[np.ndarray, None] = None
        self.grad_min: Union[np.ndarray, None] = None
        self.hess_min: Union[np.ndarray, None] = None
        self.res_min: Union[np.ndarray, None] = None
        self.sres_min: Union[np.ndarray, None] = None

    def update(self,
               x: np.ndarray,
               sensi_orders: Tuple[int],
               mode: str,
               result: ResultType) -> None:
        """Update history and best found value."""
        self.history.update(x, sensi_orders, mode, result)
        self._update_vals(x, sensi_orders, mode, result)

    def finalize(self):
        self.history.finalize()

    def _update_vals(self,
                     x: np.ndarray,
                     sensi_orders: Tuple[int],
                     mode: str,
                     result: ResultType):
        """
        Update initial and best function values.
        """
        # update initial point
        if self.fval0 is None and np.allclose(x, self.x0) \
                and 0 in sensi_orders:
            if mode == MODE_FUN:
                self.fval0 = result[FVAL]
                self.x0 = x
            else:  # mode == MODE_RES:
                chi2 = res_to_chi2(result[RES])
                self.fval0 = result.get(FVAL, chi2)
                self.x0 = x

        # update best point
        if 0 in sensi_orders:
            # extract function value
            if mode == MODE_FUN:
                fval = result[FVAL]
            else:  # mode == MODE_RES:
                chi2 = res_to_chi2(result[RES])
                fval = result.get(FVAL, chi2)
            # store value
            if fval < self.fval_min:
                self.fval_min = fval
                self.x_min = x
                self.grad_min = result.get(GRAD)
                self.hess_min = result.get(HESS)
                self.res_min = result.get(RES)
                self.sres_min = result.get(SRES)


def ndarray2string_full(x: np.ndarray):
    """
    Helper function that converts numpy arrays to string with 16 digit
    numerical precision and no truncation for large arrays

    Parameters
    ----------
    x:
        array to convert

    Returns
    -------
    x:
        array as string
    """
    if x is None:
        return None
    return np.array2string(x, threshold=len(x), precision=16,
                           max_line_width=np.inf)


def extract_values(sensi_orders: Tuple[int, ...],
                   mode: str,
                   result: ResultType,
                   options: HistoryOptions) -> Dict:
    """Extract values to record from result."""
    if mode == MODE_FUN:
        fval = np.NaN if 0 not in sensi_orders \
            else result.get(FVAL, np.NaN)
        grad = None if not options.trace_record_grad \
            or 1 not in sensi_orders \
            else result.get(GRAD, None)
        hess = None if not options.trace_record_hess \
            or 2 not in sensi_orders \
            else result.get(HESS, None)
        res = None
        sres = None
        chi2 = np.NaN
        schi2 = None
    else:  # mode == MODE_RES
        res_result = result.get(RES, None)
        sres_result = result.get(SRES, None)
        chi2 = np.NaN if not options.trace_record_chi2 \
            or 0 not in sensi_orders \
            else res_to_chi2(res_result)
        schi2 = None if not options.trace_record_schi2 \
            or 1 not in sensi_orders \
            else sres_to_schi2(res_result, sres_result)
        fval = np.NaN if 0 not in sensi_orders \
            else result.get(FVAL, chi2)
        grad = None if not options.trace_record_grad \
            or 1 not in sensi_orders \
            else schi2
        hess = None if not options.trace_record_hess \
            or 1 not in sensi_orders \
            else sres_to_fim(sres_result)
        res = None if not options.trace_record_res \
            or 0 not in sensi_orders \
            else res_result
        sres = None if not options.trace_record_sres \
            or 1 not in sensi_orders \
            else sres_result

    return {FVAL: fval, GRAD: grad, HESS: hess, RES: res, SRES: sres,
            CHI2: chi2, SCHI2: schi2}
