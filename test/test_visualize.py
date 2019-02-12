import pypesto
import pypesto.visualize
import numpy as np
import scipy as sp
import unittest


# Dfine some helping function, to have the test code more readable
def create_problem():
    # define a pypesto objective (with tracing options)
    objective_options = pypesto.ObjectiveOptions(trace_record=True,
                                                 trace_save_iter=1)
    objective = pypesto.Objective(fun=sp.optimize.rosen,
                                  grad=sp.optimize.rosen_der,
                                  hess=sp.optimize.rosen_hess,
                                  options=objective_options)

    # define a pypesto problem
    lb = -7 * np.ones((1, 2))
    ub = 7 * np.ones((1, 2))
    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    return problem


def create_optimization_result(problem):
    # write some dummy results for optimization
    result = pypesto.Result(problem=problem)
    for j in range(0, 3):
        optimizer_result = pypesto.OptimizerResult(fval=j * 0.01,
                                                   x=[j + 0.1, j + 1])
        result.optimize_result.append(optimizer_result=optimizer_result)
    for j in range(0, 4):
        optimizer_result = pypesto.OptimizerResult(fval=10 + j * 0.01,
                                                   x=[2.5 + j + 0.1,
                                                      2 + j + 1])
        result.optimize_result.append(optimizer_result=optimizer_result)

    return result


def create_optimization_history(problem):
    # create optimizer
    optimizer_options = {'maxiter': 200}
    optimizer = pypesto.ScipyOptimizer(method='TNC', options=optimizer_options)

    # run optimization
    optimize_options = pypesto.OptimizeOptions(allow_failed_starts=True)
    result_with_trace = pypesto.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=5,
        startpoint_method=pypesto.startpoint.uniform,
        options=optimize_options
    )

    return result_with_trace


def create_profile_result(result):
    # write some dummy results for profiling
    ratio_path_1 = [0.15, 0.25, 0.7, 1., 0.8, 0.35, 0.15]
    ratio_path_2 = [0.1, 0.2, 0.7, 1., 0.8, 0.3, 0.1]
    x_path_1 = np.array([[2., 2.1, 2.3, 2.5, 2.7, 2.9, 3.],
                         [1., 1.2, 1.4, 1.5, 1.6, 1.8, 2.]])
    x_path_2 = np.array([[1., 1.1, 1.3, 1.5, 1.7, 1.9, 2.1],
                         [2.1, 2.2, 2.4, 2.5, 2.8, 2.9, 3.1]])
    fval_path_1 = [4., 3., 1., 0., 1.5, 2.5, 5.]
    fval_path_2 = [4.5, 3.5, 1.5, 0., 1.3, 2.3, 4.3]
    tmp_result_1 = pypesto.ProfilerResult(x_path_1, fval_path_1, ratio_path_1)
    tmp_result_2 = pypesto.ProfilerResult(x_path_2, fval_path_2, ratio_path_2)

    # use pypesto function to write the numeric values into the results
    result.profile_result.create_new_profile_list()
    result.profile_result.create_new_profile(tmp_result_1)
    result.profile_result.create_new_profile(tmp_result_2)

    return result


def create_plotting_options():
    # create sets of reference points (from tuple, dict and from list)
    ref1 = ([1., 1.5], 0.2)
    ref2 = ([1.8, 1.9], 0.6)
    ref3 = {'x': np.array([1.4, 1.7]), 'fval': 0.4}
    ref4 = [ref1, ref2]
    ref_point = pypesto.visualize.create_references(ref4)

    # alternative figure size
    alt_fig_size = (9.0, 8.0)

    return ref1, ref2, ref3, ref4, ref_point, alt_fig_size


# create the pypesto problem
problem = create_problem()

# create some artificial optimization results, which can be visualized
result = create_optimization_result(problem)

# create some real optimization results, since we need the trace
result_with_trace = create_optimization_history(problem)

# create some options for plotting
(ref1, ref2, ref3, ref4, ref_point, alt_fig_size) = create_plotting_options()


class TestVisualize(unittest.TestCase):

    @staticmethod
    def test_waterfall():
        pypesto.visualize.waterfall(result)

    @staticmethod
    def test_waterfall_with_options():
        # Test with y-limits as vector
        pypesto.visualize.waterfall(result,
                                    reference=ref_point,
                                    y_limits=[-0.5, 2.5],
                                    size=alt_fig_size)

        # Test with y-limits as float
        pypesto.visualize.waterfall(result,
                                    reference=ref3,
                                    y_limits=5.)

    @staticmethod
    def test_waterfall_lowlevel():
        # test empty input
        pypesto.visualize.waterfall_lowlevel([])

        # test if it runs at all
        fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
        pypesto.visualize.waterfall_lowlevel(fvals)
        fvals = np.array(fvals)
        pypesto.visualize.waterfall_lowlevel(fvals)

    def test_assign_clusters(self):
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
        self.assertEqual(len(clust), 5)
        self.assertEqual(len(clustsize), 3)
        self.assertEqual(len(ind_clust), 5)

    @staticmethod
    def test_assign_clustered_colors():
        # test empty input
        pypesto.visualize.assign_clustered_colors([])

        # test if it runs at all
        fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
        pypesto.visualize.assign_clustered_colors(fvals)
        fvals = np.array(fvals)
        pypesto.visualize.assign_clustered_colors(fvals)

    @staticmethod
    def test_parameters():
        pypesto.visualize.parameters(result)

    @staticmethod
    def test_parameters_with_options():
        pypesto.visualize.parameters(result,
                                     free_indices_only=False,
                                     reference=ref_point,
                                     size=alt_fig_size)

    @staticmethod
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

    @staticmethod
    def test_profiles():
        pypesto.visualize.profiles(result)

    @staticmethod
    def test_profiles_with_options():
        pypesto.visualize.profiles(result,
                                   reference=ref_point,
                                   size=alt_fig_size)

    @staticmethod
    def test_profiles_lowlevel():
        # test empty input
        pypesto.visualize.profiles_lowlevel([])

        # test if it runs at all
        p1 = np.array([[2., 2.1, 2.3, 2.5, 2.7, 2.9, 3.],
                       [0.15, 0.25, 0.7, 1., 0.8, 0.35, 0.15]])
        p2 = np.array([[1., 1.2, 1.4, 1.5, 1.6, 1.8, 2.],
                       [0.1, 0.2, 0.5, 1., 0.6, 0.4, 0.1]])
        fvals = [p1, p2]
        pypesto.visualize.profiles_lowlevel(fvals)

    @staticmethod
    def test_profile_lowlevel():
        # test empty input
        pypesto.visualize.profile_lowlevel(fvals=[])

        # test if it runs at all
        fvals = np.array([[2., 2.1, 2.3, 2.5, 2.7, 2.9, 3.],
                          [0.15, 0.25, 0.7, 1., 0.8, 0.35, 0.15]])
        pypesto.visualize.profile_lowlevel(fvals=fvals)

    @staticmethod
    def test_reference_points(ref1, ref2, ref3, ref4):
        # Try conversion from different inputs
        pypesto.visualize.create_references(ref1)
        pypesto.visualize.create_references(references=ref2)
        ref_list_1 = pypesto.visualize.create_references(ref3)
        ref_list_2 = pypesto.visualize.create_references(ref4)

        # Try to append to a list
        ref_list_2.append(ref_list_1[0])
        ref_list_2 = pypesto.visualize.create_references(ref_list_2)

        # Try to append one point to a list via the interface
        pypesto.visualize.create_references(references=ref_list_2,
                                            x=ref2[0], fval=ref2[1])

    @staticmethod
    def test_optimizer_history():
        pypesto.visualize.optimizer_history(result_with_trace)

    @staticmethod
    def test_optimizer_history_with_options():
        # Test with y-limits as vector
        pypesto.visualize.optimizer_history(result_with_trace,
                                            y_limits=[-0.5, 2.5],
                                            reference=ref_point,
                                            size=alt_fig_size,
                                            trace_x='steps',
                                            trace_y='fval',
                                            offset_y=-10.)

        # Test with y-limits as float
        pypesto.visualize.optimizer_history(result_with_trace,
                                            y_limits=5.,
                                            reference=ref3,
                                            trace_x='time',
                                            offset_y=10.)

    @staticmethod
    def test_optimizer_history_lowlevel():
        # test empty input
        pypesto.visualize.optimizer_history([])

        # pass numpy array
        x_vals = np.array(list(range(10)))
        y_vals = 11. * np.ones(10) - x_vals
        vals1 = np.array([x_vals, y_vals])
        vals2 = np.array([0.1 * np.ones(10) + x_vals, np.ones(10) + y_vals])
        vals = [vals1, vals2]

        # test with numpy arrays
        pypesto.visualize.optimizer_history_lowlevel(vals1)
        pypesto.visualize.optimizer_history_lowlevel(vals2)

        # test with a list of arrays
        pypesto.visualize.optimizer_history_lowlevel(vals)


if __name__ == '__main__':
    unittest.main()
