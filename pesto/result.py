class Result:
    """
    Universal result object for pesto. The algorithms like optimize, profile,
    sample fill different parts of it.

    Attributes
    ----------

    problem: pesto.Problem
        The problem underlying the results.

    optimizer_results:
        The results of the optimizer runs.

    profiler_results:
        The results of the profiler run.

    sampler_results:
        The results of the sampler.

    """

    def __init__(self, problem):
        self.problem = problem
        self.optimizer_results = []
        self.profiler_results = []
        self.sampler_results = []

    def append_optimizer_result(self, optimizer_result):
        """
        Append an optimizer result to the result object.

        Parameters
        ----------

        optimizer_result:
            The result of one (local) optimizer run.
        """
        self.optimizer_results.append(optimizer_result)

    def sort_optimizer_results(self):
        """
        Sort the optimizer results by function value fval (ascending).
        """
        self.optimizer_results = sorted(self.optimizer_results,
                                        key=lambda res: res.fval)

    def get_optimizer_results_for_key(self, key) -> list:
        """
        Extract the list of values for the specified key from the optimization
        results.

        Parameters
        ----------
        key: str
            Name of the field to extract.
        """
        return [optim_res[key] for optim_res in self.optimizer_results]
