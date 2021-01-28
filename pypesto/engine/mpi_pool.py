from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
import cloudpickle as pickle
import logging
from typing import List

from .base import Engine
from .task import Task


logger = logging.getLogger(__name__)


def work(pickled_task):
    """Unpickle and execute task."""
    task = pickle.loads(pickled_task)
    return task.execute()


class MPIPoolEngine(Engine):
    """
    Parallelize the task execution using multiprocessing.

    Parameters
    ----------
    chunksize:
        The chunksize the MPIPoolExecutor uses.
        Default is two, but should always be set to
        the number of cores used in each node.
    """

    def __init__(self):
        super().__init__()

    def execute(self, tasks: List[Task]):
        """Pickle tasks and distribute work over nodes."""

        pickled_tasks = [pickle.dumps(task) for task in tasks]

        n_procs = MPI.COMM_WORLD.Get_size()   # Size of communicator
        logger.info(f"Performing parallel task execution on {n_procs-1} "
                    f"worker cores with one manager core.")
        with MPIPoolExecutor() as executor:
            results = executor.map(work, pickled_tasks)
        return results
