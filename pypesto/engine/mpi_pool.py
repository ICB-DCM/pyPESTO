from multiprocessing import Pool
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
import cloudpickle as pickle
import logging
from typing import List

from .base import Engine
from .task import Task


logger = logging.getLogger(__name__)


def work_per_node(chunk):
    with Pool() as pool:
        results = pool.map(work_per_core, chunk)
    return results


def work_per_core(pickled_task):
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

    def __init__(self, chunksize: int = 2):
        super().__init__()
        self.chunksize: int = chunksize

    def execute(self, tasks: List[Task]):
        """Pickle tasks and distribute work over nodes."""

        pickled_tasks = [pickle.dumps(task) for task in tasks]
        # put the tasks in batchsizes of the number of cores used
        chunks = [pickled_tasks[x:x+self.chunksize]
                  for x in range(0, len(pickled_tasks), self.chunksize)]

        n_procs = MPI.COMM_WORLD.Get_size()   # Size of communicator
        logger.info(f"Performing parallel task execution on {n_procs} "
                    f"nodes with chunksize of {self.chunksize}.")
        with MPIPoolExecutor() as executor:
            results_list = executor.map(work_per_node, chunks)

        results_flat = []
        for results in results_list:
            results_flat.extend(results)
        return results_flat
