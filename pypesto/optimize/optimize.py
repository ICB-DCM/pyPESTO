import numpy as np
from pypesto import Result
from .startpoint import assign_startpoints, uniform
import traceback


class OptimizeOptions(dict):
    
    def __init__(self,
                 startpoint_method=None,
                 startpoint_resample=False,
                 allow_exceptions=False,
                 tmp_save=False
                 tmp_file=None)
        super().__init__()
        
        if startpoint_method is None:
            startpoint_method = uniform
        self.startpoint_method = startpoint_method
        
        self.startpoint_resample = startpoint_resample        
        self.allow_exceptions = allow_exceptions
        self.tmp_save = tmp_save
        self.tmp_file = tmp_file
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


<<<<<<< HEAD
def minimize(problem, optimizer,
             n_starts,
             startpoint_method=uniform,
             result=None,
             startpoint_resampling=True,
             allow_failed_starts=True) -> Result:
=======
def minimize(
        problem,
        optimizer,
        n_starts,
        result=None,
        options=None) -> Result:
>>>>>>> feature_fixedpars
    """

    This is the main function to be called to perform multistart optimization.

    Parameters
    ----------

    problem: pypesto.Problem
        The problem to be solved.

    optimizer: pypesto.Optimizer
        The optimizer to be used n_starts times.

    n_starts: int
        Number of starts of the optimizer.

    startpoint_method: {callable, bool}
        Method for how to choose start points. False means the optimizer does
        not require start points.

    startpoint_resampling: bool
        Flag indicating whether initial points are supposed to be resampled if
        function evaluation fails at the initial point

    result: pypesto.Result
        A result object to append the optimization results to. For example,
        one might append more runs to a previous optimization. If None,
        a new object is created.

    allow_failed_starts: bool
        Flag indicating whether we tolerate that exceptions are thrown during
        the minimization process.


    """
    
    # check options
    if options is None:
        # use default
        options = OptimizeOptions()

    # assign startpoints
    startpoints = assign_startpoints(n_starts=n_starts,
                                     problem=problem,
                                     options=options)

    # prepare result object
    if result is None:
        result = Result(problem)

    # do multistart optimization
    for j in range(0, n_starts):
        startpoint = startpoints[j, :]
        try:
            optimizer_result = optimizer.minimize(problem, startpoint, j)
        except Exception as err:
            handle_exception(options.allow_failed_starts
<<<<<<< HEAD
            if allow_failed_starts:
                print(('start ' + str(j) + ' failed: {0}').format(err))
                optimizer_result = optimizer.recover_result(
                    problem,
                    startpoint,
                    err
                )
            else:
                raise

        result.optimize_result.append(optimizer_result=optimizer_result)
=======
            print(('start ' + str(j) + ' failed: {0}').format(err))
            traceback.print_exc()
>>>>>>> feature_fixedpars

    # sort by best fval
    result.optimize_result.sort()

    return result
   


