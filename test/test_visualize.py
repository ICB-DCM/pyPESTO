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
    # test empty input
    pypesto.visualize.waterfall_lowlevel([])

    # test if it runs at all
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
    pypesto.visualize.waterfall_lowlevel(fvals)
    fvals = np.array(fvals)
    pypesto.visualize.waterfall_lowlevel(fvals)


def test_assign_clusters():
    # test empty input
    pypesto.visualize.assign_clusters([])

    # test if it runs at all
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11, 10]
    pypesto.visualize.assign_clusters(fvals)
    fvals = np.array(fvals)
    clust, clustsize, ind_clust = pypesto.visualize.assign_clusters(fvals)

    # test if clustering works as intended
    fvals = [0., 0.00001, 1., 2., 2.001]
    clust, clustsize, ind_clust = pypesto.visualize.assign_clusters(fvals)
    assert len(clustsize) == 3


def test_assign_clustered_colors():
    # test empty input
    pypesto.visualize.assign_clustered_colors([])

    # test if it runs at all
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
    pypesto.visualize.assign_clustered_colors(fvals)
    fvals = np.array(fvals)
    pypesto.visualize.assign_clustered_colors(fvals)


def test_parameters():
    pypesto.visualize.parameters(result)


def test_parameters_lowlevel():
    # test empty input
    xs = np.array([])
    xs.shape = (0, 0)  # we can assume in input that xs.ndim == 2
    fvals = np.array([])
    pypesto.visualize.parameters_lowlevel(xs, fvals)

    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
    xs = [[0.1, 1], [1.2, 3], [2, 4], [1.2, 4.1], [1.1, 3.5],
          [4.2, 3.5], [1, 4], [6.2, 5], [4.3, 3], [3, 2]]

    # pass lists
    pypesto.visualize.parameters_lowlevel(xs, fvals, lb=lb, ub=ub)

    # pass numpy arrays
    fvals = np.array(fvals)
    xs = np.array(xs)
    pypesto.visualize.parameters_lowlevel(xs, fvals, lb=lb, ub=ub)

    # test no bounds
    pypesto.visualize.parameters_lowlevel(xs, fvals)
