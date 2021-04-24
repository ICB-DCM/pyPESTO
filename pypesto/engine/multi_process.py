from multiprocessing import Pool
import cloudpickle as pickle
import os
import logging
from typing import List
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

    def execute(self, tasks: List[Task], progress_bar: bool = True):
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
        logger.info(f"Performing parallel task execution on {n_procs} "
                    f"processes.")

        with Pool(processes=n_procs) as pool:
            results = pool.map(work,
                               tqdm(pickled_tasks,
                                    disable=not progress_bar)
                               )

        return results
