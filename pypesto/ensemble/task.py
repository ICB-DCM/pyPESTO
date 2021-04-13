import logging
import numpy as np
from typing import Any, Callable, List

from ..engine import Task

logger = logging.getLogger(__name__)


class EnsembleTask(Task):
    """Apply a method to each vector in a set of parameter vectors.

    Attributes
    ----------
    method:
        The method.
    vectors:
        The parameter vectors, with shape `(n_parameters, n_vectors)`.
    id:
        The task ID.
    """

    def __init__(
            self,
            method: Callable,
            vectors: np.ndarray,
            id: str,
    ):
        super().__init__()
        self.method = method
        self.vectors = vectors
        self.id = id

    def execute(self) -> List[Any]:
        logger.info(f"Executing task {self.id}.")
        results = []
        for index in range(self.vectors.shape[1]):
            results.append(self.method(self.vectors[:, index]))
        return results
