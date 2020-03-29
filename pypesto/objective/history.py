import numpy as np
import pandas as pd
import copy
import time
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from .constants import (
    MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES, CHI2, SCHI2, TIME,
    N_FVAL, N_GRAD, N_HESS, N_RES, N_SRES, X)
from .util import res_to_chi2, sres_to_schi2, sres_to_fim

ResultType = Dict[str, Union[float, np.ndarray]]

CSV = 'CSV'
HDF5 = 'HDF5'


class HistoryOptions(dict):
    """
    Options for the objective that are used in optimization, profiles
    and sampling.

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
    """

    def __init__(self,
                 trace_record: bool = False,
                 trace_record_grad: bool = True,
                 trace_record_hess: bool = True,
                 trace_record_res: bool = True,
                 trace_record_sres: bool = True,
                 trace_record_chi2: bool = True,
                 trace_record_schi2: bool = True,
                 trace_save_iter: int = 10):
        super().__init__()

        self.trace_record = trace_record
        self.trace_record_grad = trace_record_grad
        self.trace_record_hess = trace_record_hess
        self.trace_record_res = trace_record_res
        self.trace_record_sres = trace_record_sres
        self.trace_record_chi2 = trace_record_chi2
        self.trace_record_schi2 = trace_record_schi2
        self.trace_save_iter = trace_save_iter

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


class ObjectiveHistory:
    """Base class for objective history.
    Tracks numbers of function evaluations and maintains a history.

    Parameters
    ----------
    id:
        Identifier for the history.
    x_names:
        Parameter names.
    storage_file:
        File to save the history to. Occurrences of "{id}" in the file name
        are replaced by the `id`, if provided.
    storage_format:
        Format of the storage file.
    options:
        History options.

    Attributes
    ----------
    n_fval, n_grad, n_hess, n_res, n_sres:
        Counters of function values, gradients and Hessians,
        residuals and residual sensitivities.
    trace:
        Trace containing history of function values and parameters, if those
        were recorded.
    start_time:
        Starting time.
    """

    def __init__(self,
                 id: str = '',
                 x_names: Iterable[str] = None,
                 storage_file: str = None,
                 storage_format: str = None,
                 options: Optional[HistoryOptions] = None) -> None:
        self.id: str = id

        self.x_names: Union[List[str], None] = x_names

        self.n_fval: int = 0
        self.n_grad: int = 0
        self.n_hess: int = 0
        self.n_res: int = 0
        self.n_sres: int = 0

        self.trace: Union[pd.DataFrame, None] = None

        self.start_time = time.time()

        # translate chi2 to fval
        self._fval2chi2_offset: Union[float, None] = None

        if storage_file not in [None, CSV, HDF5]:
            raise ValueError(
                f"file {storage_file} must be in {[None, CSV, HDF5]}")
        if self.id is not None and storage_file is not None:
            storage_file = storage_file.replace("{id}", str(self.id))
        self.storage_file: str = storage_file

        if storage_format is None and storage_file is not None:
            if storage_file.endswith('csv'):
                storage_format = CSV
            elif storage_file.endswith('hdf5'):
                storage_format = HDF5
            else:
                raise ValueError(f"Cannot deduce format for {storage_file}")
        if storage_format == HDF5:
            raise ValueError(
                "Storage to HDF5 is not yet supported. Use CSV instead.")
        self.storage_format: str = storage_format

        if options is None:
            options = HistoryOptions()
        self.options: HistoryOptions = options

        # create trace file dirs
        if self.storage_file is not None:
            dirname = os.path.dirname(self.storage_file)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

    def update(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...],
            mode: str,
            result: ResultType,
            call_mode_fun: Union[Callable, None]
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
        call_mode_fun:
            Callback that may be needed to transform residuals to function
            values.
        """
        self.update_counts(sensi_orders, mode)
        self.update_trace(x, sensi_orders, mode, result, call_mode_fun)

    def finalize(self):
        """Finalize history. Called after a run."""
        self.save_trace(finalize=True)

    def fval2chi2_offset(
            self,
            x: np.ndarray,
            chi2: float,
            call_mode_fun: Union[Callable, None]):
        """
        Initializes the conversion factor between fval and chi2 values,
        if possible.
        """
        # cache
        if self._fval2chi2_offset is None:
            if call_mode_fun is not None:
                self._fval2chi2_offset = call_mode_fun(x, (0,))[FVAL] - chi2
            else:
                self._fval2chi2_offset = 0.0

        return self._fval2chi2_offset

    def update_counts(self,
                      sensi_orders: Tuple[int, ...],
                      mode: str):
        """
        Update the counters.
        """
        if mode == MODE_FUN:
            if 0 in sensi_orders:
                self.n_fval += 1
            if 1 in sensi_orders:
                self.n_grad += 1
            if 2 in sensi_orders:
                self.n_hess += 1
        elif mode == MODE_RES:
            if 0 in sensi_orders:
                self.n_res += 1
            if 1 in sensi_orders:
                self.n_sres += 1

    def init_trace(self, x: np.ndarray):
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

        self.trace = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns),
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
            self.trace[(var, float('nan'))] = \
                self.trace[(var, float('nan'))].astype(dtype)

    def update_trace(self,
                     x: np.ndarray,
                     sensi_orders: Tuple[int],
                     mode: str,
                     result: ResultType,
                     call_mode_fun: Union[Callable, None]):
        """
        Update and possibly store the trace.
        """
        if not self.options.trace_record:
            return

        # init trace
        if self.trace is None:
            self.init_trace(x)

        # extract function values
        fval, grad, hess, res, sres, chi2, schi2 = \
            self.extract_values(sensi_orders, mode, result, x, call_mode_fun)

        used_time = time.time() - self.start_time

        # create table row
        row = pd.Series(name=len(self.trace),
                        index=self.trace.columns,
                        dtype='object')

        values = {
            TIME: used_time,
            N_FVAL: self.n_fval,
            N_GRAD: self.n_grad,
            N_HESS: self.n_hess,
            N_RES: self.n_res,
            N_SRES: self.n_sres,
            FVAL: fval,
            RES: res,
            SRES: sres,
            CHI2: chi2,
            HESS: hess,
        }

        for var, val in values.items():
            row[(var, float('nan'))] = val

        for var, val in {X: x, GRAD: grad, SCHI2: schi2}.items():
            if var == X or self.options[f'trace_record_{var}']:
                row[var] = val if val is not None else np.NaN
            else:
                row[(var, float('nan'))] = None

        self.trace = self.trace.append(row)

        # save trace to file
        self.save_trace()

    def extract_values(self, sensi_orders, mode, result, x, call_mode_fun):
        """Extract values to record from result."""
        if mode == MODE_FUN:
            fval = np.NaN if 0 not in sensi_orders \
                else result.get(FVAL, np.NaN)
            grad = None if not self.options.trace_record_grad \
                or 1 not in sensi_orders \
                else result.get(GRAD, None)
            hess = None if not self.options.trace_record_hess \
                or 2 not in sensi_orders \
                else result.get(HESS, None)
            res = None
            sres = None
            chi2 = np.NaN
            schi2 = None
        else:  # mode == MODE_RES
            res_result = result.get(RES, None)
            sres_result = result.get(SRES, None)
            chi2 = np.NaN if not self.options.trace_record_chi2 \
                or 0 not in sensi_orders \
                else res_to_chi2(res_result)
            schi2 = None if not self.options.trace_record_schi2 \
                or 1 not in sensi_orders \
                else sres_to_schi2(res_result, sres_result)
            chi2_offset = self.fval2chi2_offset(x, chi2, call_mode_fun)
            fval = np.NaN if 0 not in sensi_orders \
                else chi2 + chi2_offset
            grad = None if not self.options.trace_record_grad \
                or 1 not in sensi_orders \
                else schi2
            hess = None if not self.options.trace_record_hess \
                or 1 not in sensi_orders \
                else sres_to_fim(sres_result)
            res = None if not self.options.trace_record_res \
                or 0 not in sensi_orders \
                else res_result
            sres = None if not self.options.trace_record_sres \
                or 1 not in sensi_orders \
                else sres_result

        return fval, grad, hess, res, sres, chi2, schi2

    def save_trace(self, finalize: bool = False):
        """
        Save to file via pd.DataFrame.to_csv() if `self.storage_file` is
        not None and other conditions apply.

        .. note::
            Format might be revised when storage is implemented.
        """
        if self.storage_file is None:
            return

        if finalize \
                or (len(self.trace) > 0 and len(self.trace) %
                    self.options.trace_save_iter == 0):
            # save
            trace_copy = copy.deepcopy(self.trace)
            for field in [('hess', np.NaN), ('res', np.NaN), ('sres', np.NaN)]:
                trace_copy[field] = trace_copy[field].apply(
                    ndarray2string_full
                )
            trace_copy.to_csv(self.storage_file)


class OptimizerHistory(ObjectiveHistory):
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
                 id: str,
                 x0: np.ndarray,
                 x_names: Optional[Iterable[str]] = None,
                 storage_file: str = None,
                 storage_format: str = None,
                 options: Optional[HistoryOptions] = None) -> None:
        super().__init__(id=id, x_names=x_names,
                         storage_file=storage_file,
                         storage_format=storage_format,
                         options=options)

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
               result: ResultType,
               call_mode_fun: Union[Callable, None]) -> None:
        """Update history and best found value."""
        super().update(x, sensi_orders, mode, result, call_mode_fun)
        self.update_vals(x, sensi_orders, mode, result, call_mode_fun)

    def update_vals(self,
                    x: np.ndarray,
                    sensi_orders: Tuple[int],
                    mode: str,
                    result: ResultType,
                    call_mode_fun: Union[Callable, None]):
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
                chi2_offset = self.fval2chi2_offset(x, chi2, call_mode_fun)
                self.fval0 = chi2 + chi2_offset
                self.x0 = x

        # update best point
        if 0 in sensi_orders:
            # extract function value
            if mode == MODE_FUN:
                fval = result[FVAL]
            else:  # mode == MODE_RES:
                chi2 = res_to_chi2(result[RES])
                chi2_offset = self.fval2chi2_offset(x, chi2, call_mode_fun)
                fval = chi2 + chi2_offset
            # store value
            if fval <= self.fval_min:
                self.fval_min = fval
                self.x_min = x
                self.grad_min = result.get(GRAD)
                self.hess_min = result.get(HESS)
                self.res_min = result.get(RES)
                self.sres_min = result.get(SRES)


class OptimizerHistoryFactory:
    """Factory for :class:`OptimizerHistory` objects.

    Parameters
    ----------
    storage_file:
        File to save the history to. Occurrences of "{id}" in the file name
        are replaced by the `id`, if provided.
    storage_format:
        Format of the storage file.
    options:
        Options for the history.
    """

    def __init__(self,
                 storage_file: str = None,
                 storage_format: str = None,
                 options: Dict = None):
        self.storage_file = storage_file
        self.storage_format = storage_format
        self.options = options

    def create(
            self, id: str, x0: np.ndarray, x_names: Iterable[str]
    ) -> OptimizerHistory:
        """Factory method creating a :class:`OptimizerHistory` object.
        Called for each multistart separately, just before the start.

        Parameters
        ----------
        id:
            Identifier for the history.
        x_names:
            Parameter names.
        """
        return OptimizerHistory(
            id=id, x0=x0, x_names=x_names,
            storage_file=self.storage_file,
            storage_format=self.storage_format,
            options=self.options)


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
