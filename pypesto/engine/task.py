class Task:

    def __init__(self):
        """
        Create a task object.
        """
        pass

    def execute(self):
        """
        Execute the task and return its results.
        """
        return NotImplementedError(
            "This is a non-functional base class.")


class OptimizerTask(Task):

    def __init__(self, optimizer, problem, startpoint, j_start,
                 options, handle_exception):
        super().__init__()

        self.optimizer = optimizer
        self.problem = problem
        self.startpoint = startpoint
        self.j_start = j_start
        self.options = options
        self.handle_exception = handle_exception

    def execute(self):
        try:
            optimizer_result = self.optimizer.minimize(
                self.problem, self.startpoint, self.j_start)
        except Exception as err:
            if self.options.allow_failed_starts:
                optimizer_result = self.handle_exception(
                    self.problem.objective, self.startpoint, self.j_start,
                    err)
            else:
                raise

        return optimizer_result
