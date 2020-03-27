from concurrent.futures import ThreadPoolExecutor
import copy
import os
import logging
from typing import List

from .base import Engine
from .task import Task


logger = logging.getLogger(__name__)


def work(task):
    """Just execute task."""
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

    def __init__(self, n_threads: int = None):
        super().__init__()

        if n_threads is None:
            n_threads = os.cpu_count()
            logger.warning(
                f"Engine set up to use up to {n_threads} processes in total. "
                f"The number was automatically determined and might not be "
                f"appropriate on some systems.")
        self.n_threads: int = n_threads

    def execute(self, tasks: List[Task]):
        """Deepcopy tasks and distribute work over parallel threads."""
        n_tasks = len(tasks)

        copied_tasks = [copy.deepcopy(task) for task in tasks]

        n_threads = min(self.n_threads, n_tasks)
        logger.info(f"Performing parallel task execution on {n_threads} "
                    f"threads.")
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            results = pool.map(work, copied_tasks)

        return results
