from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
import cloudpickle as pickle
import logging
from tqdm import tqdm

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
    Parallelize the task execution using
    `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_.

    To be called with:
    ``mpiexec -np #Workers+1 python -m mpi4py.futures YOURFILE.py``
    """

    def __init__(self):
        super().__init__()

    def execute(self, tasks: List[Task], progress_bar: bool = True):
        """Pickle tasks and distribute work to workers.

        Parameters
        ----------
        tasks:
            List of tasks to execute.
        progress_bar:
            Whether to display a progress bar.
        """

        pickled_tasks = [pickle.dumps(task) for task in tasks]

        n_procs = MPI.COMM_WORLD.Get_size()   # Size of communicator
        logger.info(f"Performing parallel task execution on {n_procs-1} "
                    f"workers with one manager.")

        with MPIPoolExecutor() as executor:
            results = executor.map(work,
                                   tqdm(pickled_tasks,
                                        disable=not progress_bar)
                                   )
        return results
