import numpy as np
from pypesto import Result
from .startpoint import assign_startpoints, uniform


class OptimizeOptions(dict):
	"""
	Options for the multistart optimization.
	
	Parameters
	----------
	
	startpoint_method: {callable, False}, optional
        Method for how to choose start points. False means the optimizer does
        not require start points.

    startpoint_resample: bool, optional
        Flag indicating whether initial points are supposed to be resampled if
        function evaluation fails at the initial point
        
	allow_failed_starts: bool, optional
        Flag indicating whether we tolerate that exceptions are thrown during
        the minimization process.
	"""
    
    def __init__(self,
                 startpoint_method=None,
                 startpoint_resample=False,
                 allow_exceptions=False)
        super().__init__()
        
        if startpoint_method is None:
            startpoint_method = uniform
        self.startpoint_method = startpoint_method
        
        self.startpoint_resample = startpoint_resample        
        self.allow_exceptions = allow_exceptions
        
        if objective_history is None:
			objective_history = ObjectiveHistory()
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def minimize(
        problem,
        optimizer,
        n_starts,
        result=None,
        options=None) -> Result:
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

    result: pypesto.Result
        A result object to append the optimization results to. For example,
        one might append more runs to a previous optimization. If None,
        a new object is created.

	options: pypesto.OptimizeOptions, optional
		Various options applied to the multistart optimization.
    """
    
    # check options
    if options is None:
        # use default
        options = OptimizeOptions()

    # assign startpoints
    startpoints = assign_startpoints(n_starts=n_starts,
                                     problem=problem,
                                     options=options)

    # prepare result
    if result is None:
        result = Result(problem)

    # do multistart optimization
    for j in range(0, n_starts):
        startpoint = startpoints[j, :]
               
        # apply optimizer
        try:
            optimizer_result = optimizer.minimize(problem, startpoint, j)
        except Exception as err:
            optimizer_result = handle_exception(
                objective=objective,
                startpoint=startpoint,
                allow_failed_starts=options.allow_failed_starts, 
                err=err)
                                                
        # append to result
        result.optimize_result.append(optimizer_result=optimizer_result)

    # sort by best fval
    result.optimize_result.sort()

    return result


def handle_exception(
		objective,
		startpoint,
		allow_failed_starts,
		err) -> OptimizerResult:
	"""
	Handle exceptions. Raise exception if allow_faile_starts is False,
	otherwise return a dummy pypesto.OptimizerResult.
	"""
    if allow_failed_starts:
        print(('start ' + str(j) + ' failed: {0}').format(err))
        optimizer_result = optimizer.recover_result(
            objective=objective, startpoint=startpoint, err=err)
        return optimizer_result
    raise err
