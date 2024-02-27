"""Engines without parallelization."""

from typing import Any

from ..util import tqdm
from .base import Engine
from .task import Task


class SingleCoreEngine(Engine):
    """
    Dummy engine for sequential execution on one core.

    .. note:: The objective itself may be multithreaded.
    """

    def __init__(self):
        super().__init__()

    def execute(
        self, tasks: list[Task], progress_bar: bool = None
    ) -> list[Any]:
        """Execute all tasks in a simple for loop sequentially.

        Parameters
        ----------
        tasks:
            List of tasks to execute.
        progress_bar:
            Whether to display a progress bar.

        Returns
        -------
        A list of results.
        """
        results = []
        for task in tqdm(
            tasks,
            enable=progress_bar,
        ):
            results.append(task.execute())

        return results
