"""Engines with multi-threading parallelization."""

import copy
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union

from ..util import tqdm
from .base import Engine
from .task import Task

logger = logging.getLogger(__name__)


def work(task):
    """Execute task."""
    return task.execute()


class MultiThreadEngine(Engine):
    """
    Parallelize the task execution using multithreading.

    Parameters
    ----------
    n_threads:
        The maximum number of threads to use in parallel.
        Defaults to the number of CPUs available on the system according to
        `os.cpu_count()`.
        The effectively used number of threads will be the minimum of
        `n_threads` and the number of tasks submitted.
    """

    def __init__(self, n_threads: Union[int, None] = None):
        super().__init__()

        if n_threads is None:
            n_threads = os.cpu_count()
            logger.info(
                f"Engine will use up to {n_threads} threads (= CPU count)."
            )
        self.n_threads: int = n_threads

    def execute(
        self, tasks: list[Task], progress_bar: bool = None
    ) -> list[Any]:
        """Deepcopy tasks and distribute work over parallel threads.

        Parameters
        ----------
        tasks:
            List of tasks to execute.
        progress_bar:
            Whether to display a progress bar.

        Returns
        -------
        A list of results.
        """
        n_tasks = len(tasks)

        copied_tasks = [copy.deepcopy(task) for task in tasks]

        n_threads = min(self.n_threads, n_tasks)
        logger.debug(f"Parallelizing on {n_threads} threads.")

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            results = list(
                tqdm(
                    pool.map(work, copied_tasks),
                    total=len(copied_tasks),
                    enable=progress_bar,
                ),
            )

        return results
