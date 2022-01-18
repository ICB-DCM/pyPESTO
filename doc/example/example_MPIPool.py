"""This file serves as an example how to use MPIPoolEngine
to optimize across nodes and also as a test for the
MPIPoolEngine."""
import numpy as np
import scipy as sp

import pypesto
import pypesto.optimize as optimize

# you need to manually import the MPIPoolEninge
from pypesto.engine.mpi_pool import MPIPoolEngine
from pypesto.store import OptimizationResultHDF5Writer, ProblemHDF5Writer


def setup_rosen_problem(n_starts: int = 2):
    """Set up the rosenbrock problem and return
    a pypesto.Problem"""
    objective = pypesto.Objective(
        fun=sp.optimize.rosen,
        grad=sp.optimize.rosen_der,
        hess=sp.optimize.rosen_hess,
    )

    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))

    # fixing startpoints
    startpoints = pypesto.startpoint.latin_hypercube(
        n_starts=n_starts, lb=lb, ub=ub
    )
    problem = pypesto.Problem(
        objective=objective, lb=lb, ub=ub, x_guesses=startpoints
    )
    return problem


# set all your code into this if condition.
# This way only one core performs the code
# and distributes the work of the optimization.
if __name__ == '__main__':
    # set number of starts
    n_starts = 2
    # create problem
    problem = setup_rosen_problem()
    # create optimizer
    optimizer = optimize.FidesOptimizer(verbose=0)

    # result is the way to call the optimization with MPIPoolEngine.
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=n_starts,
        engine=MPIPoolEngine(),
        filename=None,
        progress_bar=False,
    )

    # saving optimization results to hdf5
    file_name = 'temp_result.h5'
    opt_result_writer = OptimizationResultHDF5Writer(file_name)
    problem_writer = ProblemHDF5Writer(file_name)
    problem_writer.write(problem)
    opt_result_writer.write(result)
