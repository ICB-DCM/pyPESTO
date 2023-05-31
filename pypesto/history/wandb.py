import time
from typing import Literal, Optional, Tuple

import numpy as np

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
from .base import CountHistory, add_fun_from_res, reduce_result_via_options
from .options import HistoryOptions
from .util import ResultDict

try:
    import wandb
except ImportError:
    raise ImportError(
        "Using a wandb history requires an installation of "
        "the python package wandb. Please install wandb via "
        "`pip install wandb`."
    )

StepMetricType = Literal[N_FVAL, N_GRAD, N_HESS, N_RES, N_SRES, TIME]


class WandBHistory(CountHistory):
    """
    History class for logging to Weights&Biases.

    Notes
    -----
    Expects a `wandb.init()` call before initialization.

    Parameters
    ----------
    options:
        History options.

    step_metric:
        Metric to use for the x-axis of the plot. Defaults to C.N_FVAL.

    log_counts:
        Whether to log the number of function evaluations, gradients, residuals

    """

    def __init__(
        self,
        run_id: str,
        step_metric: Optional[StepMetricType] = None,
        options: HistoryOptions = None,
    ):
        if step_metric is None:
            step_metric = N_FVAL
        super().__init__(options=options)
        self._step_metric: StepMetricType = step_metric

        wandb.config.update({'name': run_id})
        wandb.define_metric(N_FVAL, summary="max")
        wandb.define_metric(N_GRAD, summary="max")
        wandb.define_metric(N_RES, summary="max")
        wandb.define_metric(N_SRES, summary="max")
        wandb.define_metric(TIME, summary="max")

        if options.trace_record:
            wandb.define_metric(
                FVAL, summary="min", step_metric=self._step_metric
            )
            for option, metric in (
                (options.trace_record_grad, GRAD),
                (options.trace_record_hess, HESS),
                (options.trace_record_sres, RES),
                (options.trace_record_res, SRES),
            ):
                if option:
                    wandb.define_metric(metric, step_metric=self._step_metric)

    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
        result: ResultDict,
    ):
        """See `HistoryBase` docstring."""
        # update counts
        super().update(x, sensi_orders, mode, result)
        # calculating function values from residuals
        #  and reduce via requested history options
        result: dict = reduce_result_via_options(
            add_fun_from_res(result), self.options
        )

        result[X] = x

        used_time = time.time() - self._start_time

        wand_log_dict = {
            N_FVAL: self.n_fval,
            N_GRAD: self.n_grad,
            N_HESS: self.n_hess,
            N_RES: self.n_res,
            N_SRES: self.n_sres,
            TIME: used_time,
        }

        if self.options.trace_record is not None:
            wand_log_dict.update(
                {
                    key: value
                    if np.isscalar(value)
                    else wandb.Histogram(value.flatten())
                    for key, value in result.items()
                    if not np.isnan(value).all()
                }
            )

        wandb.log(wand_log_dict)

    def finalize(
        self,
        message: str = None,
        exitflag: str = None,
    ) -> None:
        """See `HistoryBase` docstring."""
        wandb.log({"message": message, "exitflag": exitflag})
        wandb.finish()
