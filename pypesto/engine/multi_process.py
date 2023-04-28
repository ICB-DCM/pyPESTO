"""Engines with multi-process parallelization."""
import logging
import multiprocessing
import os
from typing import Any, List

import cloudpickle as pickle
from tqdm import tqdm

from .base import Engine
from .task import Task

logger = logging.getLogger(__name__)


def work(pickled_task):
    """Unpickle and execute task."""
    task = pickle.loads(pickled_task)
    return task.execute()


class MultiProcessEngine(Engine):
    """
    Parallelize the task execution using multiprocessing.

    Parameters
    ----------
    n_procs:
        The maximum number of processes to use in parallel.
        Defaults to the number of CPUs available on the system according to
        `os.cpu_count()`.
        The effectively used number of processes will be the minimum of
        `n_procs` and the number of tasks submitted.
    method:
        Start method, any of "fork", "spawn", "forkserver", or None,
        giving the system specific default context.
    """

    def __init__(self, n_procs: int = None, method: str = None):
        super().__init__()

        if n_procs is None:
            n_procs = os.cpu_count()
            logger.info(
                f"Engine will use up to {n_procs} processes (= CPU count)."
            )
        self.n_procs: int = n_procs
        self.method: str = method

    def execute(
        self, tasks: List[Task], progress_bar: bool = True
    ) -> List[Any]:
        """Pickle tasks and distribute work over parallel processes.

        Parameters
        ----------
        tasks:
            List of tasks to execute.
        progress_bar:
            Whether to display a progress bar.
        """
        n_tasks = len(tasks)

        pickled_tasks = [pickle.dumps(task) for task in tasks]

        n_procs = min(self.n_procs, n_tasks)
        logger.debug(f"Parallelizing on {n_procs} processes.")

        ctx = multiprocessing.get_context(method=self.method)

        with ctx.Pool(processes=n_procs) as pool:
            results = list(
                tqdm(
                    pool.imap(work, pickled_tasks),
                    total=len(pickled_tasks),
                    disable=not progress_bar,
                ),
            )

        return results
