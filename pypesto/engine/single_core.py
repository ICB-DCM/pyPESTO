from .base import Engine


class SingleCoreEngine(Engine):

    def __init__(self):
        pass

    def execute(self, tasks):
        result = []
        for task in tasks:
            results.append(task.execute())

        return results
