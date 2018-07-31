class Result:
    """
    Universal result object for pesto. The algorithms like optimize, profile,
    sample fill different parts of it.

    """

    def __init__(self):
        self.problem = None
        self.optimization = OptimizationResult()
        self.profiles = None
        self.sampling = None


class OptimizationResult:

        def __init__(self):
            self.n_starts = None
            self.par0 = None
            self.par = None
            self.fval0 = None
            self.fval = None
            self.grad = None
            self.hess = None
            self.n_feval = None
            self.n_iter = None
            self.t = None
            self.exitflag = None
            self.comment =
            pass

        def insert(self, optimizer_result):

        def sort(self):
            pass
