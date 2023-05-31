import time
from typing import Tuple

import numpy as np

from ..C import TIME, ModeType, ResultDict, X
from .base import CountHistory, add_fun_from_res, reduce_result_via_options

try:
    import wandb
except ImportError:
    raise ImportError(
        "Using a wandb history requires an installation of "
        "the python package wandb. Please install wandb via "
        "`pip install wandb`."
    )


class WandBHistory(CountHistory):
    """
    History class for logging to Weights&Biases.

    Notes
    -----
    Expects a `wandb.init()` call before the first update.
    """

    def update(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
        result: ResultDict,
    ):
        """See `HistoryBase` docstring."""
        # calculating function values from residuals
        #  and reduce via requested history options
        result: dict = reduce_result_via_options(
            add_fun_from_res(result), self.options
        )

        result[X] = x

        used_time = time.time() - self._start_time
        result[TIME] = used_time

        wandb.log(
            {
                key: value
                for key, value in result.items()
                if not np.isnan(value)
            }
        )

    def finalize(
        self,
        message: str = None,
        exitflag: str = None,
    ) -> None:
        """See `HistoryBase` docstring."""
        wandb.add_tags({'exitflag': exitflag})
        wandb.finish()
