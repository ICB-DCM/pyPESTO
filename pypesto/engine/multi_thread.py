from concurrent.futures import ThreadPoolExecutor
import copy
import os
import logging
from typing import List

from .base import Engine
from .task import Task


logger = logging.getLogger(__name__)


def work(task):
    return task.execute()


class MultiThreadEngine(Engine):
    """
    Parallelize the task execution using multithreading.

    Attributes
    ----------

    n_procs: int, optional
        The maximum number of threads to , unless less tasks are defined.
        Defaults to the number of cpus available on the system according to
        ``os.cpu_count()``.
    """

    def __init__(self, n_procs: int = None):
        super().__init__()

        if n_procs is None:
            n_procs = os.cpu_count()
            logger.warning(
                f"Engine set up to use up to {n_procs} processes in total. "
                f"The number was automatically determined and might not be "
                f"appropriate on some systems.")
        self.n_procs: int = n_procs

    def execute(self, tasks: List[Task]):
        n_tasks = len(tasks)

        copied_tasks = [copy.deepcopy(task) for task in tasks]
        n_procs = min(self.n_procs, n_tasks)
        logger.info(f"Performing parallel task execution on {n_procs} "
                    f"threads.")
        print(n_tasks)
        with ThreadPoolExecutor(max_workers=n_procs) as pool:
            results = pool.map(work, copied_tasks)

        return results
