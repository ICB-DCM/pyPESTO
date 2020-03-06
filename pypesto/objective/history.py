import numpy as np
import pandas as pd
import itertools as itt
import time
import os

from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS, RES, SRES
from .util import res_to_chi2, sres_to_schi2
from .options import ObjectiveOptions

from typing import List, Union, Optional, Dict, Tuple
ResultType = Dict[str, Union[float, np.ndarray]]


class ObjectiveHistory:
    """
    Objective call history. Also handles saving of intermediate results.

    Parameters
    ------------

    options: ObjectiveOptions, optional
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
    """

    def __init__(self,
                 options: Optional[ObjectiveOptions] = None,
                 x_names: Optional[List[str]] = None) -> None:

        if options is None:
            options = ObjectiveOptions()
        self.options: Union[ObjectiveOptions, None] = options
        self.x_names: Union[List[str], None] = x_names

        self.n_fval: Union[int, None] = None
        self.n_grad: Union[int, None] = None
        self.n_hess: Union[int, None] = None
        self.n_res: Union[int, None] = None
        self.n_sres: Union[int, None] = None

        self.trace: Union[List[pd.Dataframe], None] = None
        self.start_time: Union[float, None] = None

        self.fval_min: Union[float, None] = None
        self.x_min: Union[np.ndarray, None] = None
        self.fval0: Union[float, None] = None
        self.x0: Union[np.ndarray, None] = None

        self.index: Union[str, None] = None

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
               result: ResultType) -> None:
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
        """
        self._update_counts(sensi_orders, mode)
        self._update_trace(x, sensi_orders, mode, result)
        self._update_vals(x, sensi_orders, mode, result)

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

    def _init_trace(self, x: np.ndarray):
        """
        Initialize the trace.
        """
        if self.x_names is None:
            self.x_names = [f'x{i}' for i, _ in enumerate(x)]

        columns: List[Tuple] = [
            (c, np.NaN) for c in [
                'time', 'n_fval', 'n_grad', 'n_hess', 'n_res', 'n_sres',
                'fval', 'chi2', 'res', 'sres',
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
        if self.trace is None:
            self._init_trace(x)

        # extract function values
        if mode == MODE_FUN:
            fval = None if 0 not in sensi_orders \
                else result.get(FVAL, None)
            grad = None if not self.options.trace_record_grad \
                or 1 not in sensi_orders \
                else result.get(GRAD, None)
            hess = None if not self.options.trace_record_hess \
                or 2 not in sensi_orders \
                else result.get(HESS, None)
            res = None
            sres = None
            chi2 = None
            schi2 = None
        else:  # mode == MODE_RES
            fval = None
            grad = None
            hess = None
            res_result = result.get(RES, None)
            sres_result = result.get(SRES, None)
            res = None if not self.options.trace_record_res \
                or 0 not in sensi_orders \
                else res_result
            sres = None if not self.options.trace_record_sres \
                or 1 not in sensi_orders \
                else sres_result
            chi2 = None if not self.options.trace_record_chi2 \
                or 0 not in sensi_orders \
                else res_to_chi2(res_result)
            schi2 = None if not self.options.trace_record_schi2 \
                or 1 not in sensi_orders \
                else sres_to_schi2(res_result, sres_result)

        # check whether to append to trace
        if not self.options.trace_all and fval >= self.fval_min:
            return

        used_time = time.time() - self.start_time

        # create table row
        index = len(self.trace)
        self.trace.append(pd.Series(name=index, dtype='float64'))

        values = {
            'time': used_time,
            'n_fval': self.n_fval,
            'n_grad': self.n_grad,
            'n_hess': self.n_hess,
            'n_res': self.n_sres,
            'fval': fval,
            'res': res,
            'sres': sres,
            'chi2': chi2,
            'hess': hess,
        }
        for var, val in values.items():
            self.trace.loc[index, var] = val

        for var, val in {'x': x, 'grad': grad, 'schi2': schi2}.items():
            if var == 'x' or self.options[f'trace_record_{var}']:
                for ix, x_name in enumerate(self.x_names):
                    self.trace.loc[index, (var, x_name)] = val[ix] \
                        if val is not None else np.NaN
            else:
                self.trace.loc[index, var] = None

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
            self.trace.to_csv(filename)

    def _update_vals(self,
                     x: np.ndarray,
                     sensi_orders: Tuple[int],
                     mode: str,
                     result: ResultType):
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
                self.fval0 = res_to_chi2(result[RES])
                self.x0 = x

        # update best point
        fval = np.inf
        if 0 in sensi_orders:
            if mode == MODE_FUN:
                fval = result[FVAL]
            else:  # mode == MODE_RES:
                fval = res_to_chi2(result[RES])
        if fval < self.fval_min:
            self.fval_min = fval
            self.x_min = x
