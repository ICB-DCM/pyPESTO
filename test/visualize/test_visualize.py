import functools
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
import pytest
import os
import petab

import pypesto
import pypesto.petab
import pypesto.optimize as optimize
import pypesto.predict as predict
import pypesto.profile as profile
import pypesto.sample as sample
import pypesto.visualize as visualize
import pypesto.ensemble as ensemble
from pypesto.visualize.model_fit import visualize_optimized_model_fit, \
    time_trajectory_model
from functools import wraps
from typing import Sequence


def close_fig(fun):
    """Close figure."""

    @wraps(fun)
    def wrapped_fun(*args, **kwargs):
        ret = fun(*args, **kwargs)
        plt.close('all')
        return ret

    return wrapped_fun


# Define some helper functions, to have the test code more readable


def create_bounds(n_parameters: int = 2):
    # define bounds for a pypesto problem
    lb = -7 * np.ones((1, n_parameters))
    ub = 7 * np.ones((1, n_parameters))

    return lb, ub


def create_problem(n_parameters: int = 2,
                   x_names: Sequence[str] = None):
    # define a pypesto objective
    objective = pypesto.Objective(fun=so.rosen,
                                  grad=so.rosen_der,
                                  hess=so.rosen_hess,
                                  x_names=x_names)

    # define a pypesto problem
    (lb, ub) = create_bounds(n_parameters)
    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    return problem


def create_petab_problem():
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(current_path,
                                            '..', '..',
                                            'doc', 'example'))

    # import to petab
    petab_problem = petab.Problem.from_yaml(
        dir_path + "/conversion_reaction/conversion_reaction.yaml")
    # import to pypesto
    importer = pypesto.petab.PetabImporter(petab_problem)
    # create problem
    problem = importer.create_problem()

    return problem


def sample_petab_problem():
    # create problem
    problem = create_petab_problem()

    sampler = sample.AdaptiveMetropolisSampler()
    result = sample.sample(problem, n_samples=1000,
                           sampler=sampler,
                           x0=np.array([3, -4]),
                           filename=None)
    return result


def create_optimization_result(n=4):
    # create the pypesto problem
    problem = create_problem()

    # write some dummy results for optimization
    result = pypesto.Result(problem=problem)
    for j in range(0, 3):
        optimizer_result = optimize.OptimizerResult(
            id=str(j), fval=j * 0.01, x=np.array([j + 0.1, j + 1]),
            grad=np.array([2.5 + j + 0.1, 2 + j + 1]))
        result.optimize_result.append(optimizer_result=optimizer_result)
    for j in range(0, n):
        optimizer_result = optimize.OptimizerResult(
            id=str(j + 3), fval=10 + j * 0.01,
            x=np.array([2.5 + j + 0.1, 2 + j + 1]),
            grad=np.array([j + 0.1, j + 1]))
        result.optimize_result.append(optimizer_result=optimizer_result)

    return result


def create_optimization_result_nan_inf():
    """
    Create a result object containing nan and inf function values
    """
    # get result with only numbers
    result = create_optimization_result()

    # append nan and inf
    optimizer_result = optimize.OptimizerResult(
        fval=float('nan'), x=np.array([float('nan'), float('nan')]))
    result.optimize_result.append(optimizer_result=optimizer_result)
    optimizer_result = optimize.OptimizerResult(
        fval=-float('inf'), x=np.array([-float('inf'), -float('inf')]))
    result.optimize_result.append(optimizer_result=optimizer_result)

    return result


def create_optimization_history():
    # create the pypesto problem
    problem = create_problem()

    # create optimizer
    optimizer_options = {'maxiter': 200}
    optimizer = optimize.ScipyOptimizer(
        method='TNC', options=optimizer_options)

    history_options = pypesto.HistoryOptions(
        trace_record=True, trace_save_iter=1)

    # run optimization
    optimize_options = optimize.OptimizeOptions(allow_failed_starts=True)
    result_with_trace = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=5,
        startpoint_method=pypesto.startpoint.uniform,
        options=optimize_options,
        history_options=history_options,
        filename=None
    )

    return result_with_trace


def create_profile_result():
    # create a pypesto result
    result = create_optimization_result()

    # write some dummy results for profiling
    ratio_path_1 = [0.15, 0.25, 0.7, 1., 0.8, 0.35, 0.15]
    ratio_path_2 = [0.1, 0.2, 0.7, 1., 0.8, 0.3, 0.1]
    x_path_1 = np.array([[2., 2.1, 2.3, 2.5, 2.7, 2.9, 3.],
                         [1., 1.2, 1.4, 1.5, 1.6, 1.8, 2.]])
    x_path_2 = np.array([[1., 1.1, 1.3, 1.5, 1.7, 1.9, 2.1],
                         [2.1, 2.2, 2.4, 2.5, 2.8, 2.9, 3.1]])
    fval_path_1 = [4., 3., 1., 0., 1.5, 2.5, 5.]
    fval_path_2 = [4.5, 3.5, 1.5, 0., 1.3, 2.3, 4.3]
    tmp_result_1 = profile.ProfilerResult(x_path_1, fval_path_1, ratio_path_1)
    tmp_result_2 = profile.ProfilerResult(x_path_2, fval_path_2, ratio_path_2)

    # use pypesto function to write the numeric values into the results
    result.profile_result.append_empty_profile_list()
    result.profile_result.append_profiler_result(tmp_result_1)
    result.profile_result.append_profiler_result(tmp_result_2)

    return result


def create_plotting_options():
    # create sets of reference points (from tuple, dict and from list)
    ref1 = ([1., 1.5], 0.2)
    ref2 = ([1.8, 1.9], 0.6)
    ref3 = {'x': np.array([1.4, 1.7]), 'fval': 0.4}
    ref4 = [ref1, ref2]
    ref_point = visualize.create_references(ref4)

    return ref1, ref2, ref3, ref4, ref_point


def post_processor(
        amici_outputs,
        output_type,
        output_ids,
):
    """An ensemble prediction post-processor.

    This post_processor will transform the output of the simulation tool such
    that the output is compatible with other methods, such as plotting
    routines.
    """
    outputs = [
        amici_output[output_type]
        if amici_output[predict.constants.AMICI_STATUS] == 0
        else np.full(
            (len(amici_output[predict.constants.AMICI_T]), len(output_ids)),
            np.nan
        )
        for amici_output in amici_outputs
    ]
    return outputs


@close_fig
def test_waterfall_w_zoom():
    # create the necessary results
    result_1 = create_optimization_result(500)
    result_2 = create_optimization_result()

    # test a standard call
    visualize.waterfall_w_zoom(result_1, n_starts_to_zoom=10, zoom=3)

    # test plotting of lists
    visualize.waterfall_w_zoom([result_1, result_2], n_starts_to_zoom=3)


@close_fig
def test_waterfall():
    # create the necessary results
    result_1 = create_optimization_result()
    result_2 = create_optimization_result()

    # test a standard call
    visualize.waterfall(result_1)

    # test plotting of lists
    visualize.waterfall([result_1, result_2])


@close_fig
def test_waterfall_with_nan_inf():
    # create the necessary results, one with nan and inf, one without
    result_1 = create_optimization_result_nan_inf()
    result_2 = create_optimization_result()

    # test a standard call
    visualize.waterfall(result_1)

    # test plotting of lists
    visualize.waterfall([result_1, result_2])


@close_fig
def test_waterfall_with_options():
    # create the necessary results
    result_1 = create_optimization_result()
    result_2 = create_optimization_result()

    # alternative figure size and plotting options
    (_, _, ref3, _, ref_point) = create_plotting_options()
    alt_fig_size = (9.0, 8.0)

    with pytest.warns(UserWarning, match="Invalid lower bound"):
        # Test with y-limits as vector and invalid lower bound
        visualize.waterfall(result_1,
                            reference=ref_point,
                            y_limits=[-0.5, 2.5],
                            start_indices=[0, 1, 4, 11],
                            size=alt_fig_size,
                            colors=[1., .3, .3, 0.5])

    # Test with fully invalid bounds
    with pytest.warns(UserWarning, match="Invalid bounds"):
        visualize.waterfall(result_1, y_limits=[-1.5, 0.])

    # Test with y-limits as float
    with pytest.warns(UserWarning, match="Offset specified by user"):
        visualize.waterfall([result_1, result_2],
                            reference=ref3,
                            offset_y=-2.5,
                            start_indices=3,
                            y_limits=5.)

    # Test with linear scale
    visualize.waterfall(result_1,
                        reference=ref3,
                        scale_y='lin',
                        offset_y=0.2,
                        y_limits=5.)


@close_fig
def test_waterfall_lowlevel():
    # test empty input
    visualize.waterfall_lowlevel([])

    # test if it runs at all
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
    visualize.waterfall_lowlevel(fvals)
    fvals = np.array(fvals)
    visualize.waterfall_lowlevel(fvals)


@pytest.mark.parametrize("scale_to_interval", [None, (0, 1)])
@close_fig
def test_parameters(scale_to_interval):
    # create the necessary results
    result_1 = create_optimization_result()
    result_2 = create_optimization_result()

    # test a standard call
    visualize.parameters(result_1,
                         scale_to_interval=scale_to_interval)

    # test plotting of lists
    visualize.parameters([result_1, result_2],
                         scale_to_interval=scale_to_interval)


@pytest.mark.parametrize("scale_to_interval", [None, (0, 1)])
@close_fig
def test_parameters_with_nan_inf(scale_to_interval):
    # create the necessary results
    result_1 = create_optimization_result_nan_inf()
    result_2 = create_optimization_result_nan_inf()

    # test a standard call
    visualize.parameters(result_1,
                         scale_to_interval=scale_to_interval)

    # test plotting of lists
    visualize.parameters([result_1, result_2],
                         scale_to_interval=scale_to_interval)


@pytest.mark.parametrize("scale_to_interval", [None, (0, 1)])
@close_fig
def test_parameters_with_options(scale_to_interval):
    # create the necessary results
    result_1 = create_optimization_result()
    result_2 = create_optimization_result()

    # alternative figure size and plotting options
    (_, _, _, _, ref_point) = create_plotting_options()
    alt_fig_size = (9.0, 8.0)

    # test calls with specific options
    visualize.parameters(result_1,
                         parameter_indices='all',
                         reference=ref_point,
                         size=alt_fig_size,
                         colors=[1., .3, .3, 0.5],
                         scale_to_interval=scale_to_interval)

    visualize.parameters([result_1, result_2],
                         parameter_indices='all',
                         reference=ref_point,
                         balance_alpha=False,
                         start_indices=(0, 1, 4),
                         scale_to_interval=scale_to_interval)

    visualize.parameters([result_1, result_2],
                         parameter_indices='free_only',
                         start_indices=3,
                         scale_to_interval=scale_to_interval)


@close_fig
def test_parameters_lowlevel():
    # create some dummy results
    (lb, ub) = create_bounds()
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
    xs = [[0.1, 1], [1.2, 3], [2, 4], [1.2, 4.1], [1.1, 3.5],
          [4.2, 3.5], [1, 4], [6.2, 5], [4.3, 3], [3, 2]]

    # pass lists
    visualize.parameters_lowlevel(xs, fvals, lb=lb, ub=ub)

    # pass numpy arrays
    fvals = np.array(fvals)
    xs = np.array(xs)
    visualize.parameters_lowlevel(xs, fvals, lb=lb, ub=ub)

    # test no bounds
    visualize.parameters_lowlevel(xs, fvals)


@close_fig
def test_parameters_hist():
    # create the pypesto problem
    problem = create_problem()

    # create optimizer
    optimizer_options = {'maxiter': 200}
    optimizer = optimize.ScipyOptimizer(
        method='TNC', options=optimizer_options)

    # run optimization
    result_1 = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=10,
        startpoint_method=pypesto.startpoint.uniform,
        filename=None
    )

    visualize.parameter_hist(result_1, 'x1')
    visualize.parameter_hist(result_1, 'x1', start_indices=list(range(10)))


# @close_fig
def test_ensemble_dimension_reduction():
    # creates a test problem
    problem = create_problem(n_parameters=20)

    # =========================================================================
    # test ensemble identifiability if some bounds are hit and some aren't
    my_ensemble = []
    # some magical numbers which create a reasonable plot. Please don't change!
    std = (1, 1, 2, 2, 2.5, 3, 3, 4, 5, 7, 6, 6, 10, 8, 10, 15, 15, 25, 35, 50)
    offset = (1, 1, 0, 0, -1, -1, -7, -7, 5, 4,
              -10, -10, -12, 0, 1, -13, 0, -15, -18, -20)
    # create a collection/an ensemble based on these magic numbers
    for ip in range(len(std)):
        my_ensemble.append(std[ip] * np.random.rand(100) + offset[ip])
    my_ensemble = ensemble.Ensemble(np.array(my_ensemble),
                                    lower_bound=problem.lb,
                                    upper_bound=problem.ub)

    # test plotting from a collection object
    umap_components, umap_embedding = \
        ensemble.get_umap_representation_parameters(my_ensemble,
                                                    n_components=3)

    # test call via high-level routine
    visualize.projection_scatter_umap(umap_components, components=(0, 1, 2))

    # test call via low-level routine 1
    visualize.ensemble_crosstab_scatter_lowlevel(
        umap_components, component_labels=('A', 'B', 'C'))

    pca_components, pca_object = \
        ensemble.get_pca_representation_parameters(
            my_ensemble, rescale_data=True, n_components=6)

    # test call via high-level routine
    visualize.projection_scatter_pca(pca_components, components=range(4))
    visualize.projection_scatter_pca(pca_components)

    # test call via lowlevel routine
    visualize.ensemble_scatter_lowlevel(pca_components[:, 0:2])


@close_fig
def test_ensemble_identifiability():
    # creates a test problem
    problem = create_problem(n_parameters=100)

    # =========================================================================
    # test ensemble identifiability if some bounds are hit and some aren't
    my_ensemble = []
    # some magical numbers which create a reasonable plot. Please don't change!
    std = (1, 1, 2, 2, 2.5, 3, 3, 4, 5, 7, 6, 6, 10, 8, 10, 15, 15, 25, 35, 50)
    offset = (1, 1, 0, 0, -1, -1, -7, -7, 5, 4,
              -10, -10, -12, 0, 1, -13, 0, -15, -18, -20)
    # create a collection/an ensemble based on these magic numbers
    for _ in range(5):
        for ip in range(len(std)):
            my_ensemble.append(std[ip] * np.random.rand(500) + offset[ip])
    my_ensemble = np.array(my_ensemble)
    my_ensemble = ensemble.Ensemble(my_ensemble,
                                    lower_bound=problem.lb,
                                    upper_bound=problem.ub)

    # test plotting from a collection object
    visualize.ensemble_identifiability(my_ensemble)

    # =========================================================================
    # test ensemble identifiability if no bounds are hit
    # create an ensemble within tight bounds
    my_ensemble = [
        (1 + np.cos(ix) ** 2) * np.random.rand(500) - 1. + np.sin(ix)
        for ix in range(100)]
    my_ensemble = ensemble.Ensemble(np.array(my_ensemble),
                                    lower_bound=problem.lb,
                                    upper_bound=problem.ub)

    # test plotting from a collection object
    visualize.ensemble_identifiability(my_ensemble)


@close_fig
def test_profiles():
    # create the necessary results
    result_1 = create_profile_result()
    result_2 = create_profile_result()

    # test a standard call
    visualize.profiles(result_1)

    # test plotting of lists
    visualize.profiles([result_1, result_2])


@close_fig
def test_profiles_with_options():
    # create the necessary results
    result = create_profile_result()
    result.profile_result.list.append([result.profile_result.list[0][1], None])

    # alternative figure size and plotting options
    (_, _, _, _, ref_point) = create_plotting_options()
    alt_fig_size = (9.0, 8.0)

    # test a call with some specific options
    visualize.profiles(result,
                       reference=ref_point,
                       size=alt_fig_size,
                       profile_list_ids=[0, 1],
                       legends=['profile list 0', 'profile list 1'],
                       colors=[[1., .3, .3, .5], [.5, .9, .4, .3]])


@close_fig
def test_profiles_lowlevel():
    # test empty input
    visualize.profiles_lowlevel([])

    # test if it runs at all using dummy results
    p1 = np.array([[2., 2.1, 2.3, 2.5, 2.7, 2.9, 3.],
                   [0.15, 0.25, 0.7, 1., 0.8, 0.35, 0.15]])
    p2 = np.array([[1., 1.2, 1.4, 1.5, 1.6, 1.8, 2.],
                   [0.1, 0.2, 0.5, 1., 0.6, 0.4, 0.1]])
    fvals = [p1, p2]
    visualize.profiles_lowlevel(fvals)


@close_fig
def test_profile_lowlevel():
    # test empty input
    visualize.profile_lowlevel(fvals=[])

    # test if it runs at all using dummy results
    fvals = np.array([[2., 2.1, 2.3, 2.5, 2.7, 2.9, 3.],
                      [0.15, 0.25, 0.7, 1., 0.8, 0.35, 0.15]])
    visualize.profile_lowlevel(fvals=fvals)


@close_fig
def test_profile_cis():
    """Test the profile approximate confidence interval visualization."""
    result = create_profile_result()
    visualize.profile_cis(result, confidence_level=0.99)
    visualize.profile_cis(
        result, show_bounds=True, profile_indices=[0])


@close_fig
def test_optimizer_history():
    # create the necessary results
    result_1 = create_optimization_history()
    result_2 = create_optimization_history()

    # test a standard call
    visualize.optimizer_history(result_1)

    # test plotting of lists
    visualize.optimizer_history([result_1,
                                 result_2])


@close_fig
def test_optimizer_history_with_options():
    # create the necessary results
    result_1 = create_optimization_history()
    result_2 = create_optimization_history()

    # alternative figure size and plotting options
    (_, _, ref3, _, ref_point) = create_plotting_options()
    alt_fig_size = (9.0, 8.0)

    # Test with y-limits as vector
    with pytest.warns(UserWarning, match="Invalid lower bound"):
        visualize.optimizer_history(result_1,
                                    y_limits=[-0.5, 2.5],
                                    start_indices=[0, 1, 4, 11],
                                    reference=ref_point,
                                    size=alt_fig_size,
                                    trace_x='steps',
                                    trace_y='fval',
                                    offset_y=-10.,
                                    colors=[1., .3, .3, 0.5])

    # Test with linear scale
    visualize.optimizer_history([result_1,
                                 result_2],
                                y_limits=[-0.5, 2.5],
                                start_indices=[0, 1, 4, 11],
                                reference=ref_point,
                                size=alt_fig_size,
                                scale_y='lin')

    # Test with y-limits as float
    visualize.optimizer_history(result_1,
                                y_limits=5.,
                                start_indices=3,
                                reference=ref3,
                                trace_x='time',
                                trace_y='gradnorm',
                                offset_y=10.)


@close_fig
def test_optimizer_history_lowlevel():
    # test empty input
    visualize.optimizer_history_lowlevel([])

    # pass numpy array
    x_vals = np.array(list(range(10)))
    y_vals = 11. * np.ones(10) - x_vals
    vals1 = np.array([x_vals, y_vals])
    vals2 = np.array([0.1 * np.ones(10) + x_vals, np.ones(10) + y_vals])
    vals = [vals1, vals2]

    # test with numpy arrays
    visualize.optimizer_history_lowlevel(vals1)
    visualize.optimizer_history_lowlevel(vals2)

    # test with a list of arrays
    visualize.optimizer_history_lowlevel(vals)


@close_fig
def test_optimization_stats():
    """ Test pypesto.visualize.optimization_stats """

    # create the pypesto problem
    problem = create_problem()

    # create optimizer
    optimizer_options = {'maxiter': 200}
    optimizer = optimize.ScipyOptimizer(
        method='TNC', options=optimizer_options)

    # run optimization
    result_1 = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=10,
        startpoint_method=pypesto.startpoint.uniform,
        filename=None
    )

    result_2 = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=10,
        startpoint_method=pypesto.startpoint.uniform,
        filename=None
    )

    visualize.optimization_run_property_per_multistart(result_1, 'n_fval',
                                                       legends='best result')

    visualize.optimization_run_property_per_multistart(result_1, 'n_fval',
                                                       plot_type='hist',
                                                       legends='best result')

    visualize.optimization_run_property_per_multistart(result_2, 'n_fval')

    # test plotting of lists
    visualize.optimization_run_property_per_multistart(
        [result_1, result_2], 'n_fval',
        legends=['result1', 'result2'],
        plot_type='line')

    visualize.optimization_run_property_per_multistart(
        result_1, 'time', plot_type='hist', legends='best result')

    visualize.optimization_run_property_per_multistart(
        [result_1, result_2],
        'time',
        colors=[[.5, .9, .9, .3], [.9, .7, .8, .5]],
        legends=['result1', 'result2'],
        plot_type='hist')

    visualize.optimization_run_properties_per_multistart([result_1, result_2])

    visualize.optimization_run_properties_one_plot(result_1, ['time'])

    visualize.optimization_run_properties_one_plot(result_1, ['n_fval',
                                                              'n_grad',
                                                              'n_hess'])

    visualize.optimization_run_property_per_multistart(
        [result_1, result_2],
        'time',
        colors=[[.5, .9, .9, .3], [.9, .7, .8, .5]],
        legends=['result1', 'result2'],
        plot_type='both')


@close_fig
def test_optimize_convergence():
    result = create_optimization_result()
    result_nan = create_optimization_result_nan_inf()

    visualize.optimizer_convergence(result)
    visualize.optimizer_convergence(result_nan)


def test_assign_clusters():
    # test empty input
    visualize.assign_clusters([])

    # test if it runs at all
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11, 10]
    visualize.assign_clusters(fvals)
    fvals = np.array(fvals)
    clust, clustsize = visualize.assign_clusters(fvals)
    np.testing.assert_array_equal(clust, [0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5])
    np.testing.assert_array_equal(clustsize, [2, 1, 3, 1, 3, 1])

    # test if clustering works as intended
    fvals = [0., 0.00001, 1., 2., 2.001]
    clust, clustsize = visualize.assign_clusters(fvals)
    assert len(clust) == 5
    assert len(clustsize) == 3
    np.testing.assert_array_equal(clust, [0, 0, 1, 2, 2])
    np.testing.assert_array_equal(clustsize, [2, 1, 2])


def test_assign_clustered_colors():
    # test empty input
    visualize.assign_clustered_colors([])

    # test if it runs at all
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
    visualize.assign_clustered_colors(fvals)
    fvals = np.array(fvals)
    visualize.assign_clustered_colors(fvals)


def test_assign_colors():
    # test empty input
    visualize.assign_colors([])

    # test if it runs at all
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11]
    visualize.assign_colors(fvals)
    fvals = np.array(fvals)
    visualize.assign_colors(fvals)
    fvals = [0.01, 0.02, 1.]
    visualize.assign_colors(fvals, colors=[.5, .9, .9, .3])
    visualize.assign_colors(fvals, colors=[[.5, .9, .9, .3],
                                           [.5, .8, .8, .5],
                                           [.9, .1, .1, .1]])


def test_delete_nan_inf():
    # create fvals containing nan and inf
    fvals = np.array([42, 1.5, np.nan, 67.01, np.inf])

    # create a random x
    x = np.array([[1, 2], [1, 1], [np.nan, 1], [65, 1], [2, 3]])
    x, fvals = visualize.delete_nan_inf(fvals, x)

    # test if the nan and inf in fvals are deleted, and so do the
    # corresponding entries in x
    np.testing.assert_array_equal(fvals, [42, 1.5, 67.01])
    np.testing.assert_array_equal(x, [[1, 2], [1, 1], [65, 1]])


def test_reference_points():
    # create the necessary results
    (ref1, ref2, ref3, ref4, _) = create_plotting_options()

    # Try conversion from different inputs
    visualize.create_references(ref1)
    visualize.create_references(references=ref2)
    ref_list_1 = visualize.create_references(ref3)
    ref_list_2 = visualize.create_references(ref4)

    # Try to append to a list
    ref_list_2.append(ref_list_1[0])
    ref_list_2 = visualize.create_references(ref_list_2)

    # Try to append one point to a list via the interface
    visualize.create_references(references=ref_list_2,
                                x=ref2[0], fval=ref2[1])


def test_process_result_list():
    # create the necessary results
    result_1 = create_optimization_result()
    result_2 = create_optimization_result()

    # Test empty arguments
    visualize.process_result_list([])

    # Test single argument
    # Test single argument
    visualize.process_result_list(result_1)

    # Test handling of a real list
    res_list = [result_1]
    visualize.process_result_list(res_list)
    res_list.append(result_2)
    visualize.process_result_list(res_list)


def create_sampling_result():
    """Create a result object containing sample results."""
    result = create_optimization_result()
    n_chain = 2
    n_iter = 100
    n_par = len(result.optimize_result.get_for_key('x')[0])
    trace_neglogpost = np.random.randn(n_chain, n_iter)
    trace_neglogprior = np.zeros((n_chain, n_iter))
    trace_x = np.random.randn(n_chain, n_iter, n_par)
    betas = np.array([1, .1])
    sample_result = sample.McmcPtResult(
        trace_neglogpost=trace_neglogpost,
        trace_neglogprior=trace_neglogprior,
        trace_x=trace_x, betas=betas,
        burn_in=10)
    result.sample_result = sample_result

    return result


@close_fig
def test_sampling_fval_traces():
    """Test pypesto.visualize.sampling_fval_traces"""
    result = create_sampling_result()
    visualize.sampling_fval_traces(result)
    # call with custom arguments
    visualize.sampling_fval_traces(
        result, i_chain=1, stepsize=5, size=(10, 10))


@close_fig
def test_sampling_parameter_traces():
    """Test pypesto.visualize.sampling_parameter_traces"""
    result = create_sampling_result()
    visualize.sampling_parameter_traces(result)
    # call with custom arguments
    visualize.sampling_parameter_traces(
        result, i_chain=1, stepsize=5, size=(10, 10),
        use_problem_bounds=False)


@close_fig
def test_sampling_scatter():
    """Test pypesto.visualize.sampling_scatter"""
    result = create_sampling_result()
    visualize.sampling_scatter(result)
    # call with custom arguments
    visualize.sampling_scatter(
        result, i_chain=1, stepsize=5, size=(10, 10))


@close_fig
def test_sampling_1d_marginals():
    """Test pypesto.visualize.sampling_1d_marginals"""
    result = create_sampling_result()
    visualize.sampling_1d_marginals(result)
    # call with custom arguments
    visualize.sampling_1d_marginals(
        result, i_chain=1, stepsize=5, size=(10, 10))
    # call with other modes
    visualize.sampling_1d_marginals(result, plot_type='hist')
    visualize.sampling_1d_marginals(
        result, plot_type='kde', bw='silverman')


@close_fig
def test_sampling_parameter_cis():
    """Test pypesto.visualize.sampling_parameter_cis"""
    result = create_sampling_result()
    visualize.sampling_parameter_cis(result)
    # call with custom arguments
    visualize.sampling_parameter_cis(
        result, alpha=[99, 68], step=0.1, size=(10, 10))


@close_fig
def test_sampling_prediction_trajectories():
    """Test pypesto.visualize.sampling_prediction_trajectories"""
    credibility_interval_levels = [99, 68]
    result = sample_petab_problem()
    post_processor_amici_x = functools.partial(
        post_processor,
        output_type=predict.constants.AMICI_X,
        output_ids=result.problem.objective.amici_model.getStateIds(),
    )
    predictor = predict.AmiciPredictor(
        result.problem.objective,
        post_processor=post_processor_amici_x,
        output_ids=result.problem.objective.amici_model.getStateIds(),
    )

    sample_ensemble = ensemble.Ensemble.from_sample(
        result,
        x_names=result.problem.x_names,
        ensemble_type=ensemble.EnsembleType.sample,
        lower_bound=result.problem.lb,
        upper_bound=result.problem.ub,
    )

    ensemble_prediction = sample_ensemble.predict(
        predictor,
        prediction_id=predict.constants.AMICI_X,
    )

    # Plot by
    visualize.sampling_prediction_trajectories(
        ensemble_prediction,
        levels=credibility_interval_levels,
        groupby=predict.constants.CONDITION,
    )
    visualize.sampling_prediction_trajectories(
        ensemble_prediction,
        levels=credibility_interval_levels,
        size=(10, 10),
        groupby=predict.constants.OUTPUT,
    )


@close_fig
def test_visualize_optimized_model_fit():
    """Test pypesto.visualize.visualize_optimized_model_fit"""
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(current_path,
                                            '..', '..',
                                            'doc', 'example'))

    # import to petab
    petab_problem = petab.Problem.from_yaml(
        os.path.join(dir_path, "conversion_reaction",
                     "conversion_reaction.yaml"))
    # import to pypesto
    importer = pypesto.petab.PetabImporter(petab_problem)
    # create problem
    problem = importer.create_problem()

    result = optimize.minimize(problem=problem,
                               n_starts=1,
                               filename=None)

    # test call of visualize_optimized_model_fit
    visualize_optimized_model_fit(petab_problem=petab_problem,
                                  result=result)


@close_fig
def test_time_trajectory_model():
    """Test pypesto.visualize.time_trajectory_model"""
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(current_path,
                                            '..', '..',
                                            'doc', 'example'))

    # import to petab
    petab_problem = petab.Problem.from_yaml(
        os.path.join(dir_path, "conversion_reaction",
                     "conversion_reaction.yaml"))
    # import to pypesto
    importer = pypesto.petab.PetabImporter(petab_problem)
    # create problem
    problem = importer.create_problem()

    result = optimize.minimize(problem=problem,
                               n_starts=1,
                               filename=None)

    # test call of time_trajectory_model
    time_trajectory_model(result=result)
