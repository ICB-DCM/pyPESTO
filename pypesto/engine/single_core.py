from typing import List
from tqdm import tqdm

from .base import Engine
from .task import Task


class SingleCoreEngine(Engine):
    """
    Dummy engine for sequential execution on one core. Note that the
    objective itself may be multithreaded.
    """

    def __init__(self):
        super().__init__()

    def execute(self, tasks: List[Task], progress_bar: bool = True):
        """Execute all tasks in a simple for loop sequentially.

        Parameters
        ----------
        tasks:
            List of tasks to execute.
        progress_bar:
            Whether to display a progress bar.
        """
        results = []
        for task in tqdm(tasks, disable=not progress_bar):
            results.append(task.execute())

        return results
