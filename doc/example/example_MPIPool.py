import pypesto
import numpy as np
import scipy as sp
import pypesto.optimize as optimize
# you need to manually import the MPIPoolEninge
from pypesto.engine.mpi_pool import MPIPoolEngine
# the below is needed for testing purposes.
from numpy.testing import assert_almost_equal

# set all your code into this if condition.
# This way only one core performs the code
# and distributes the work of the optimization.
if __name__ == '__main__':
    # first type of objective
    objective1 = pypesto.Objective(fun=sp.optimize.rosen,
                                   grad=sp.optimize.rosen_der,
                                   hess=sp.optimize.rosen_hess)

    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))

    # set number of starts
    n_starts = 2

    # fixing startpoints
    startpoints = pypesto.startpoint.latin_hypercube(n_starts=n_starts,
                                                     lb=lb,
                                                     ub=ub)
    problem1 = pypesto.Problem(objective=objective1, lb=lb, ub=ub,
                               x_guesses=startpoints)

    x = np.arange(-2, 2, 0.1)
    y = np.arange(-2, 2, 0.1)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    for j in range(0, x.shape[0]):
        for k in range(0, x.shape[1]):
            z[j, k] = objective1([x[j, k], y[j, k]], (0,))

    # create different optimizers
    optimizer = optimize.FidesOptimizer()

    # save optimizer trace
    history_options = pypesto.HistoryOptions(trace_record=True)

    # result2 is the way to call the optimization with MPIPoolEngine.
    result1 = optimize.minimize(
            problem=problem1, optimizer=optimizer,
            n_starts=n_starts, engine=pypesto.engine.MultiProcessEngine())
    print('\n done with MultiProcessEngine \n')
    result2 = optimize.minimize(
            problem=problem1, optimizer=optimizer,
            n_starts=n_starts, engine=MPIPoolEngine())
    print('i am done with optimization')

    # starting here are the tests (not needed in your code)
    if(result1.optimize_result.list[0]['id'] ==
            result2.optimize_result.list[0]['id']):
        assert_almost_equal(result1.optimize_result.list[0]['x'],
                            result2.optimize_result.list[0]['x'],
                            err_msg='The final parameter values '
                                    'do not agree for the engines.')
        assert_almost_equal(result1.optimize_result.list[1]['x'],
                            result2.optimize_result.list[1]['x'],
                            err_msg='The final parameter values '
                                    'do not agree for the engines.')
    else:
        assert_almost_equal(result1.optimize_result.list[0]['x'],
                            result2.optimize_result.list[1]['x'],
                            err_msg='The final parameter values '
                                    'do not agree for the engines.')
        assert_almost_equal(result1.optimize_result.list[1]['x'],
                            result2.optimize_result.list[0]['x'],
                            err_msg='The final parameter values '
                                    'do not agree for the engines.')
    print('also done with tests')
