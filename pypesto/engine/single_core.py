from typing import List

from .base import Engine
from .task import Task


class SingleCoreEngine(Engine):
    """
    Dummy engine for sequential execution on one core. Note that the
    objective itself may be multithreaded.
    """

    def __init__(self):
        super().__init__()

    def execute(self, tasks: List[Task]):
        """
        Execute all tasks in a simple for loop sequentially.
        """
        results = []
        for task in tasks:
            results.append(task.execute())

        return results
