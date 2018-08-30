import pypesto
import pypesto.visualize
import numpy as np
import scipy as sp


objective = pypesto.Objective(fun=sp.optimize.rosen,
                              grad=sp.optimize.rosen_der,
                              hess=sp.optimize.rosen_hess)
lb = -7 * np.ones((1, 2))
ub = 7 * np.ones((1, 2))
problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)
result = pypesto.Result(problem=problem)
for j in range(0, 3):
    optimizer_result = pypesto.OptimizerResult(fval=j * 0.01,
                                               x=[j + 0.1, j + 1])
    result.optimize_result.append(optimizer_result=optimizer_result)
for j in range(0, 4):
    optimizer_result = pypesto.OptimizerResult(fval=10 + j * 0.01,
                                               x=[2.5 + j + 0.1, 2 + j + 1])
    result.optimize_result.append(optimizer_result=optimizer_result)


def test_waterfall():
    pypesto.visualize.waterfall(result)


def test_waterfall_lowlevel():
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
    pypesto.visualize.waterfall_lowlevel(fvals)
    fvals = np.array(fvals)
    pypesto.visualize.waterfall_lowlevel(fvals)


def test_cluster():
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
    pypesto.visualize.get_cluster(fvals)
    fvals = np.array(fvals)
    pypesto.visualize.get_cluster(fvals)
    # TODO: evaluate calculated clusters


def test_assign_color():
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
    pypesto.visualize.assign_color(fvals)
    fvals = np.array(fvals)
    pypesto.visualize.assign_color(fvals)


def test_parameters():
    pypesto.visualize.parameters(result)


def test_parameters_lowlevel():
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
    xs = [[0.1, 1], [1.2, 3], [2, 4], [1.2, 4.1], [1.1, 3.5],
          [4.2, 3.5], [1, 4], [6.2, 5], [4.3, 3], [3, 2]]

    # pass lists
    pypesto.visualize.parameters_lowlevel(xs, fvals, lb=lb, ub=ub)

    # pass numpy arrays
    fvals = np.array(fvals)
    xs = np.array(xs)
    pypesto.visualize.parameters_lowlevel(xs, fvals, lb=lb, ub=ub)
