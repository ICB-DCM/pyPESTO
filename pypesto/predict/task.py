import logging
from collections.abc import Sequence

from ..C import ModeType
from ..engine import Task

logger = logging.getLogger(__name__)


class PredictorTask(Task):
    """Perform a single prediction with :class:`pypesto.engine.Task`.

    Designed for use with :class:`pypesto.ensemble.Ensemble`.

    Attributes
    ----------
    predictor:
        The predictor to use.
    x:
        The parameter vector to compute predictions with.
    sensi_orders:
        Specifies which sensitivities to compute, e.g. (0,1) -> fval, grad.
    mode:
        Whether to compute function values or residuals.
    id:
        The input ID.
    """

    def __init__(
        self,
        predictor,  #: 'pypesto.predict.Predictor',  # noqa: F821
        x: Sequence[float],
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        id: str,
    ):
        super().__init__()
        self.predictor = predictor
        self.x = x
        self.sensi_orders = sensi_orders
        self.mode = mode
        self.id = id

    def execute(self):  # -> 'pypesto.predict.PredictionResult':  # noqa: F821
        """Execute and return the prediction."""
        logger.debug(f"Executing task {self.id}.")
        prediction = self.predictor(self.x, self.sensi_orders, self.mode)
        return prediction
