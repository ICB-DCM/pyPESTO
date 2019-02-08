from multiprocessing import Pool
import cloudpickle as pickle
import os

from .base import Engine


def work(pickled_task):
    task = pickle.loads(pickled_task)
    return task.execute()


class MultiProcessEngine(Engine):

    def __init__(self, n_procs=None):
        if n_procs is None:
            n_procs = os.cpu_count()
            # TODO: Issue warning that this might be not safe
            # on cluster environments
        self.n_procs = n_procs

    def execute(self, tasks):
        n_tasks = len(tasks)

        pickled_tasks = []
        for task in tasks:
            pickled_tasks.append(pickle.dumps(task))

        n_procs = min(self.n_procs, n_tasks)

        with Pool(processes=n_procs) as pool:
            results = pool.map(work, pickled_tasks)

        return results
