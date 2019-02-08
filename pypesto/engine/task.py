class Task:

    def __init__(self):
        """
        Create a task object. A task is one of a list of independent
        execution tasks that are submitted to the execution engine
        to be executed using the execute() method, commonly in parallel.
        """
        pass

    def execute(self):
        """
        Execute the task and return its results.
        """
        return NotImplementedError(
            "This is a non-functional base class.")


class OptimizerTask(Task):
    """
    A multistart optimization task, performed in `pypesto.minimize`.
    """

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
