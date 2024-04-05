import logging
from typing import Any, Callable

import numpy as np

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

    def execute(self) -> list[Any]:
        """Execute the task."""
        logger.debug(f"Executing task {self.id}.")
        results = []
        for index in range(self.vectors.shape[1]):
            results.append(self.method(self.vectors[:, index]))
        return results
