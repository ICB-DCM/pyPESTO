import numpy as np
import pandas as pd
import copy
import time
import os

from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES
from .util import res_to_chi2, sres_to_schi2, sres_to_fim
from .options import ObjectiveOptions

from typing import List, Union, Optional, Dict, Tuple, Callable
ResultType = Dict[str, Union[float, np.ndarray]]


class ObjectiveHistory:
    """
    Objective call history. Also handles saving of intermediate results.

    Parameters
    ------------

    options:
        Values needed for creating a history are extracted.

    Attributes
    ----------

    n_fval, n_grad, n_hess, n_res, n_sres:
        Counters of function values, gradients and hessians,
        residuals and residual sensitivities.

    trace:
        List containing history of function values and parameters if
        options.tr_record is True.

    start_time:
        Reference start time.

    fval0, fval_min:
        Initial and best function value found.

    x0, x_min:
        Initial and best parameters found.

    x_names:
        parameter names

    index:
        Id identifying the history object when called in a multistart
        setting.

    fval2chi2_offset:
        conversion constant to convert chi2 values to fvals
    """

    def __init__(self,
                 options: Optional[ObjectiveOptions] = None,
                 x_names: Optional[List[str]] = None,
                 obj: Optional[Callable] = None) -> None:

        if options is None:
            options = ObjectiveOptions()
        self.options: Union[ObjectiveOptions, None] = options
        self.x_names: Union[List[str], None] = x_names

        self.n_fval: Union[int, None] = None
        self.n_grad: Union[int, None] = None
        self.n_hess: Union[int, None] = None
        self.n_res: Union[int, None] = None
        self.n_sres: Union[int, None] = None

        self.trace: pd.Dataframe = None

        self.start_time: Union[float, None] = None

        self.fval_min: Union[float, None] = None
        self.x_min: Union[np.ndarray, None] = None
        self.fval0: Union[float, None] = None
        self.x0: Union[np.ndarray, None] = None

        self.index: Union[str, None] = None

        self.fval2chi2_offset: Union[float, None] = None

        self.reset()

    def reset(self, index: str = None) -> None:
        """
        Reset all counters, the trace, and start the timer, and create
        directory for trace file.
        """
        self.n_fval = 0
        self.n_grad = 0
        self.n_hess = 0
        self.n_res = 0
        self.n_sres = 0

        self.trace = None
        self.start_time = time.time()

        self.fval0 = None
        self.x0 = None
        self.fval_min = np.inf
        self.x_min = None

        self.index = index

        # create trace file dirs
        if self.options.trace_file is not None:
            dirname = os.path.dirname(self.options.trace_file)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

    def update(self,
               x: np.ndarray,
               sensi_orders: Tuple[int],
               mode: str,
               result: ResultType,
               call_mode_fun: Callable) -> None:
        """
        Update the history.

        Parameters
        ----------
        x:
            The current parameter.
        sensi_orders:
            The sensitivity orders.
        mode:
            As in constants.MODE_.
        result:
            The result for x.
        call_mode_fun:
            The objective function's `_call_mode_fun` method.
            May be needed once throughout the history.
        """
        self._update_counts(sensi_orders, mode)
        self._update_trace(x, sensi_orders, mode, result, call_mode_fun)
        self._update_vals(x, sensi_orders, mode, result, call_mode_fun)

    def finalize(self):
        """
        Save the trace to file if options.trace_save is True.
        """
        self._save_trace(finalize=True)

    def _update_counts(self,
                       sensi_orders: Tuple[int],
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

    def _fval2chi2_offset(
            self, x: np.ndarray, chi2: float, call_mode_fun: Callable):
        """
        Initializes the conversion factor between fval and chi2 values,
        if possible.
        """
        if self.fval2chi2_offset is None:
            if call_mode_fun is not None:
                self.fval2chi2_offset = call_mode_fun(x, (0,))[FVAL] - chi2
            else:
                self.fval2chi2_offset = 0.0

        return self.fval2chi2_offset

    def _init_trace(self, x: np.ndarray):
        """
        Initialize the trace.
        """
        if self.x_names is None:
            self.x_names = [f'x{i}' for i, _ in enumerate(x)]

        columns: List[Tuple] = [
            (c, float('nan')) for c in [
                'time', 'n_fval', 'n_grad', 'n_hess', 'n_res', 'n_sres',
                'fval', 'chi2', 'res', 'sres', 'hess',
            ]
        ]

        for var in ['x', 'grad', 'schi2']:
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
            'res': 'object',
            'sres': 'object',
            'hess': 'object',
            'n_fval': 'int64',
            'n_grad': 'int64',
            'n_hess': 'int64',
            'n_res': 'int64',
            'n_sres': 'int64',
        }

        for var, dtype in trace_dtypes.items():
            self.trace[(var, float('nan'))] = \
                self.trace[(var, float('nan'))].astype(dtype)

    def _update_trace(self,
                      x: np.ndarray,
                      sensi_orders: Tuple[int],
                      mode: str,
                      result: ResultType,
                      call_mode_fun: Callable):
        """
        Update and possibly store the trace.
        """

        if not self.options.trace_record:
            return

        # init trace
        if self.trace is None:
            self._init_trace(x)

        # extract function values
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
            chi2_offset = self._fval2chi2_offset(x, chi2, call_mode_fun)
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

        # check whether to append to trace
        if not self.options.trace_all and fval > self.fval_min:
            return

        used_time = time.time() - self.start_time

        # create table row
        row = pd.Series(name=len(self.trace),
                        index=self.trace.columns,
                        dtype='object')

        values = {
            'time': used_time,
            'n_fval': self.n_fval,
            'n_grad': self.n_grad,
            'n_hess': self.n_hess,
            'n_res': self.n_res,
            'n_sres': self.n_sres,
            'fval': fval,
            'res': res,
            'sres': sres,
            'chi2': chi2,
            'hess': hess,
        }
        for var, val in values.items():
            row[(var, float('nan'))] = val

        for var, val in {'x': x, 'grad': grad, 'schi2': schi2}.items():
            if var == 'x' or self.options[f'trace_record_{var}']:
                row[var] = val if val is not None else np.NaN
            else:
                row[(var, float('nan'))] = None

        self.trace = self.trace.append(row)

        # save trace to file
        self._save_trace()

    def _save_trace(self, finalize: bool = False):
        """
        Save to file via pd.DataFrame.to_csv() if options.trace_file is
        not None and other conditions apply.
        Format might be revised when storage is implemented.
        """
        if self.options.trace_file is None:
            return

        if finalize \
           or (len(self.trace) > 0 and len(self.trace) %
               self.options.trace_save_iter == 0):
            filename = self.options.trace_file
            if self.index is not None:
                filename = filename.replace("{index}", str(self.index))
            # save
            trace_copy = copy.deepcopy(self.trace)
            for field in [('hess', np.NaN), ('res', np.NaN), ('sres', np.NaN)]:
                trace_copy[field] = trace_copy[field].apply(
                    ndarray2string_full
                )
            trace_copy.to_csv(filename)

    def _update_vals(self,
                     x: np.ndarray,
                     sensi_orders: Tuple[int],
                     mode: str,
                     result: ResultType,
                     call_mode_fun: Callable):
        """
        Update initial and best function values. Must be called after
        update_trace().
        """

        # update initial point
        if self.fval0 is None and 0 in sensi_orders:
            if mode == MODE_FUN:
                self.fval0 = result[FVAL]
                self.x0 = x
            else:  # mode == MODE_RES:
                chi2 = res_to_chi2(result[RES])
                chi2_offset = self._fval2chi2_offset(x, chi2, call_mode_fun)
                self.fval0 = chi2 + chi2_offset
                self.x0 = x

        # update best point
        fval = np.inf
        if 0 in sensi_orders:
            if mode == MODE_FUN:
                fval = result[FVAL]
            else:  # mode == MODE_RES:
                chi2 = res_to_chi2(result[RES])
                chi2_offset = self._fval2chi2_offset(x, chi2, call_mode_fun)
                fval = chi2 + chi2_offset
        if fval < self.fval_min:
            self.fval_min = fval
            self.x_min = x


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
