import functools
import logging
import os
from collections.abc import Sequence
from copy import deepcopy
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import petab.v1 as petab
import pytest
import scipy.optimize as so

import pypesto
import pypesto.ensemble as ensemble
import pypesto.optimize as optimize
import pypesto.petab
import pypesto.predict as predict
import pypesto.sample as sample
import pypesto.util
import pypesto.visualize as visualize
from pypesto.testing.examples import (
    get_Boehm_JProteomeRes2014_hierarchical_petab_corrected_bounds,
)
from pypesto.visualize.model_fit import (
    time_trajectory_model,
    visualize_optimized_model_fit,
)


def close_fig(fun):
    """Close figure."""

    @wraps(fun)
    def wrapped_fun(*args, **kwargs):
        ret = fun(*args, **kwargs)
        plt.close("all")
        return ret

    return wrapped_fun


# Define some helper functions, to have the test code more readable


def create_bounds(n_parameters: int = 2):
    # define bounds for a pypesto problem
    lb = -7 * np.ones((1, n_parameters))
    ub = 7 * np.ones((1, n_parameters))

    return lb, ub


def create_problem(n_parameters: int = 2, x_names: Sequence[str] = None):
    # define a pypesto objective
    objective = pypesto.Objective(
        fun=so.rosen, grad=so.rosen_der, hess=so.rosen_hess, x_names=x_names
    )

    # define a pypesto problem
    (lb, ub) = create_bounds(n_parameters)
    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    return problem


def create_petab_problem():
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(
        os.path.join(current_path, "..", "..", "doc", "example")
    )

    # import to petab
    petab_problem = petab.Problem.from_yaml(
        dir_path + "/conversion_reaction/conversion_reaction.yaml"
    )
    # import to pypesto
    importer = pypesto.petab.PetabImporter(petab_problem)
    # create problem
    problem = importer.create_problem()

    return problem


def sample_petab_problem():
    # create problem
    problem = create_petab_problem()

    sampler = sample.AdaptiveMetropolisSampler(
        options={
            "show_progress": False,
        },
    )
    result = sample.sample(
        problem,
        n_samples=1000,
        sampler=sampler,
        x0=np.array([3, -4]),
    )
    return result


def create_optimization_result(n=4):
    # create the pypesto problem
    problem = create_problem()

    # write some dummy results for optimization
    result = pypesto.Result(problem=problem)
    for k in range(0, 3):
        optimizer_result = pypesto.OptimizerResult(
            id=str(k),
            fval=k * 0.01,
            x=np.array([k + 0.1, k + 1]),
            grad=np.array([2.5 + k + 0.1, 2 + k + 1]),
        )
        result.optimize_result.append(optimize_result=optimizer_result)
    for k in range(0, n):
        optimizer_result = pypesto.OptimizerResult(
            id=str(k + 3),
            fval=10 + k * 0.01,
            x=np.array([2.5 + k + 0.1, 2 + k + 1]),
            grad=np.array([k + 0.1, k + 1]),
        )
        result.optimize_result.append(optimize_result=optimizer_result)

    return result


def create_optimization_result_nan_inf():
    """
    Create a result object containing nan and inf function values
    """
    # get result with only numbers
    result = create_optimization_result()

    # append nan and inf
    # depending on optimizer failed starts's x can be None or vector of np.nan
    optimizer_result = pypesto.OptimizerResult(
        fval=float("nan"), x=np.array([float("nan"), float("nan")]), id="nan"
    )
    result.optimize_result.append(optimize_result=optimizer_result)
    optimizer_result = pypesto.OptimizerResult(
        fval=float("nan"), x=None, id="nan_none"
    )
    result.optimize_result.append(optimize_result=optimizer_result)
    optimizer_result = pypesto.OptimizerResult(
        fval=-float("inf"),
        x=np.array([-float("inf"), -float("inf")]),
        id="inf",
    )
    result.optimize_result.append(optimize_result=optimizer_result)

    return result


def create_optimization_history():
    # create the pypesto problem
    problem = create_problem()

    # create optimizer
    optimizer_options = {"maxfun": 200}
    optimizer = optimize.ScipyOptimizer(
        method="TNC", options=optimizer_options
    )

    history_options = pypesto.HistoryOptions(
        trace_record=True, trace_save_iter=1
    )

    # run optimization
    optimize_options = optimize.OptimizeOptions(allow_failed_starts=True)
    result_with_trace = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=5,
        options=optimize_options,
        history_options=history_options,
        progress_bar=False,
    )

    return result_with_trace


def create_profile_result():
    # create a pypesto result
    result = create_optimization_result()

    # write some dummy results for profiling
    ratio_path_1 = np.array([0.15, 0.25, 0.7, 1.0, 0.8, 0.35, 0.15])
    ratio_path_2 = np.array([0.1, 0.2, 0.7, 1.0, 0.8, 0.3, 0.1])
    x_path_1 = np.array(
        [
            [2.0, 2.1, 2.3, 2.5, 2.7, 2.9, 3.0],
            [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0],
        ]
    )
    x_path_2 = np.array(
        [
            [1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1],
            [2.1, 2.2, 2.4, 2.5, 2.8, 2.9, 3.1],
        ]
    )
    fval_path_1 = [4.0, 3.0, 1.0, 0.0, 1.5, 2.5, 5.0]
    fval_path_2 = [4.5, 3.5, 1.5, 0.0, 1.3, 2.3, 4.3]
    tmp_result_1 = pypesto.ProfilerResult(x_path_1, fval_path_1, ratio_path_1)
    tmp_result_2 = pypesto.ProfilerResult(x_path_2, fval_path_2, ratio_path_2)

    # use pypesto function to write the numeric values into the results
    result.profile_result.append_empty_profile_list()
    result.profile_result.append_profiler_result(tmp_result_1)
    result.profile_result.append_profiler_result(tmp_result_2)

    return result


def create_plotting_options():
    # create sets of reference points (from tuple, dict and from list)
    ref1 = ([1.0, 1.5], 0.2)
    ref2 = ([1.8, 1.9], 0.6)
    ref3 = {"x": np.array([1.4, 1.7]), "fval": 0.4}
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
        (
            amici_output[output_type]
            if amici_output[pypesto.C.AMICI_STATUS] == 0
            else np.full(
                (len(amici_output[pypesto.C.AMICI_T]), len(output_ids)), np.nan
            )
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
    visualize.waterfall(result_1, n_starts_to_zoom=10)

    # test plotting of lists
    visualize.waterfall([result_1, result_2], n_starts_to_zoom=3)


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

    # test all-non-finite
    result_no_finite = deepcopy(result_1)
    result_no_finite.optimize_result.list = [
        or_
        for or_ in result_no_finite.optimize_result.list
        if not np.isfinite(or_.fval)
    ]
    visualize.waterfall(result_no_finite)


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
        visualize.waterfall(
            result_1,
            reference=ref_point,
            y_limits=[-0.5, 2.5],
            start_indices=[0, 1, 4, 11],
            size=alt_fig_size,
            colors=[1.0, 0.3, 0.3, 0.5],
        )

    # Test with fully invalid bounds
    with pytest.warns(UserWarning, match="Invalid bounds"):
        visualize.waterfall(result_1, y_limits=[-1.5, 0.0])

    # Test with y-limits as float
    with pytest.warns(UserWarning, match="Offset specified by user"):
        visualize.waterfall(
            [result_1, result_2],
            reference=ref3,
            offset_y=-2.5,
            start_indices=3,
            y_limits=5.0,
        )

    # Test with linear scale
    visualize.waterfall(
        result_1, reference=ref3, scale_y="lin", offset_y=0.2, y_limits=5.0
    )


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
    visualize.parameters(result_1, scale_to_interval=scale_to_interval)

    # test plotting of lists
    visualize.parameters(
        [result_1, result_2], scale_to_interval=scale_to_interval
    )


@pytest.mark.parametrize("scale_to_interval", [None, (0, 1)])
@close_fig
def test_parameters_with_nan_inf(scale_to_interval):
    # create the necessary results
    result_1 = create_optimization_result_nan_inf()
    result_2 = create_optimization_result_nan_inf()

    # test a standard call
    visualize.parameters(result_1, scale_to_interval=scale_to_interval)

    # test plotting of lists
    visualize.parameters(
        [result_1, result_2], scale_to_interval=scale_to_interval
    )


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
    visualize.parameters(
        result_1,
        parameter_indices="all",
        reference=ref_point,
        size=alt_fig_size,
        colors=[1.0, 0.3, 0.3, 0.5],
        scale_to_interval=scale_to_interval,
    )

    visualize.parameters(
        [result_1, result_2],
        parameter_indices="all",
        reference=ref_point,
        balance_alpha=False,
        start_indices=(0, 1, 4),
        scale_to_interval=scale_to_interval,
    )

    visualize.parameters(
        [result_1, result_2],
        parameter_indices="free_only",
        start_indices=3,
        scale_to_interval=scale_to_interval,
    )


@close_fig
def test_parameters_lowlevel():
    # create some dummy results
    (lb, ub) = create_bounds()
    fvals = np.array([0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11])
    xs = np.array(
        [
            [0.1, 1],
            [1.2, 3],
            [2, 4],
            [1.2, 4.1],
            [1.1, 3.5],
            [4.2, 3.5],
            [1, 4],
            [6.2, 5],
            [4.3, 3],
            [3, 2],
        ]
    )

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
    optimizer_options = {"maxfun": 200}
    optimizer = optimize.ScipyOptimizer(
        method="TNC", options=optimizer_options
    )

    # run optimization
    result_1 = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=10,
        progress_bar=False,
    )

    visualize.parameter_hist(result_1, "x1")
    visualize.parameter_hist(result_1, "x1", start_indices=list(range(10)))


@pytest.mark.parametrize("scale_to_interval", [None, (0, 1)])
@close_fig
def test_parameters_hierarchical(scale_to_interval):
    # obtain a petab problem with hierarchical parameters
    petab_problem = (
        get_Boehm_JProteomeRes2014_hierarchical_petab_corrected_bounds()
    )
    importer = pypesto.petab.PetabImporter(petab_problem, hierarchical=True)
    helper_problem = importer.create_problem()

    # set x_guesses to nominal values for fast optimization
    x_guesses = np.asarray(petab_problem.x_nominal_scaled)[
        helper_problem.x_free_indices
    ]
    problem = importer.create_problem(x_guesses=[x_guesses])

    # run optimization
    n_starts = 1
    result = optimize.minimize(
        problem=problem,
        n_starts=n_starts,
        progress_bar=False,
    )

    # test a call with hierarchical parameters
    visualize.parameters(
        result, scale_to_interval=scale_to_interval, size=(15, 12)
    )


@close_fig
def test_optimization_scatter():
    result = create_optimization_result()
    visualize.optimization_scatter(result)


@close_fig
def test_optimization_scatter_with_x_None():
    result = create_optimization_result()
    # create an optimizerResult with x=None
    optimizer_result = pypesto.OptimizerResult(x=None, fval=np.inf, id="inf")
    result.optimize_result.append(optimize_result=optimizer_result)

    visualize.optimization_scatter(result)


# @close_fig
def _test_ensemble_dimension_reduction():
    # creates a test problem
    problem = create_problem(n_parameters=20)

    # =========================================================================
    # test ensemble identifiability if some bounds are hit and some aren't
    my_ensemble = []
    # some magical numbers which create a reasonable plot. Please don't change!
    std = (1, 1, 2, 2, 2.5, 3, 3, 4, 5, 7, 6, 6, 10, 8, 10, 15, 15, 25, 35, 50)
    offset = (
        1,
        1,
        0,
        0,
        -1,
        -1,
        -7,
        -7,
        5,
        4,
        -10,
        -10,
        -12,
        0,
        1,
        -13,
        0,
        -15,
        -18,
        -20,
    )
    # create a collection/an ensemble based on these magic numbers
    for ip in range(len(std)):
        my_ensemble.append(std[ip] * np.random.rand(100) + offset[ip])
    my_ensemble = ensemble.Ensemble(
        np.array(my_ensemble), lower_bound=problem.lb, upper_bound=problem.ub
    )

    # test plotting from a collection object
    (
        umap_components,
        umap_embedding,
    ) = ensemble.get_umap_representation_parameters(
        my_ensemble, n_components=3
    )

    # test call via high-level routine
    visualize.projection_scatter_umap(umap_components, components=(0, 1, 2))

    # test call via low-level routine 1
    visualize.ensemble_crosstab_scatter_lowlevel(
        umap_components, component_labels=("A", "B", "C")
    )

    pca_components, pca_object = ensemble.get_pca_representation_parameters(
        my_ensemble, rescale_data=True, n_components=6
    )

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
    offset = (
        1,
        1,
        0,
        0,
        -1,
        -1,
        -7,
        -7,
        5,
        4,
        -10,
        -10,
        -12,
        0,
        1,
        -13,
        0,
        -15,
        -18,
        -20,
    )
    # create a collection/an ensemble based on these magic numbers
    for _ in range(5):
        for ip in range(len(std)):
            my_ensemble.append(std[ip] * np.random.rand(500) + offset[ip])
    my_ensemble = np.array(my_ensemble)
    my_ensemble = ensemble.Ensemble(
        my_ensemble, lower_bound=problem.lb, upper_bound=problem.ub
    )

    # test plotting from a collection object
    visualize.ensemble_identifiability(my_ensemble)

    # =========================================================================
    # test ensemble identifiability if no bounds are hit
    # create an ensemble within tight bounds
    my_ensemble = [
        (1 + np.cos(ix) ** 2) * np.random.rand(500) - 1.0 + np.sin(ix)
        for ix in range(100)
    ]
    my_ensemble = ensemble.Ensemble(
        np.array(my_ensemble), lower_bound=problem.lb, upper_bound=problem.ub
    )

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
    visualize.profiles(
        result,
        reference=ref_point,
        size=alt_fig_size,
        profile_list_ids=[0, 1],
        legends=["profile list 0", "profile list 1"],
        colors=[[1.0, 0.3, 0.3, 0.5], [0.5, 0.9, 0.4, 0.3]],
    )


@close_fig
def test_profiles_lowlevel():
    # test empty input
    visualize.profiles_lowlevel([])

    # test if it runs at all using dummy results
    p1 = np.array(
        [
            [2.0, 2.1, 2.3, 2.5, 2.7, 2.9, 3.0],
            [0.15, 0.25, 0.7, 1.0, 0.8, 0.35, 0.15],
        ]
    )
    p2 = np.array(
        [
            [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0],
            [0.1, 0.2, 0.5, 1.0, 0.6, 0.4, 0.1],
        ]
    )
    fvals = [p1, p2]
    visualize.profiles_lowlevel(fvals)


@close_fig
def test_profile_lowlevel():
    # test empty input
    visualize.profile_lowlevel(fvals=[])

    # test if it runs at all using dummy results
    fvals = np.array(
        [
            [2.0, 2.1, 2.3, 2.5, 2.7, 2.9, 3.0],
            [0.15, 0.25, 0.7, 1.0, 0.8, 0.35, 0.15],
        ]
    )
    visualize.profile_lowlevel(fvals=fvals)


@close_fig
def test_profile_cis():
    """Test the profile approximate confidence interval visualization."""
    result = create_profile_result()
    visualize.profile_cis(result, confidence_level=0.99)
    visualize.profile_cis(result, show_bounds=True, profile_indices=[0])


@close_fig
def test_nested_profile_cis():
    """Test the profile approximate confidence interval visualization."""
    result = create_profile_result()
    visualize.profile_nested_cis(result, confidence_levels=[0.99, 0.95, 0.9])
    visualize.profile_nested_cis(result, colors=["#5F9ED1", "#007ACC"])


@close_fig
def test_optimizer_history():
    # create the necessary results
    result_1 = create_optimization_history()
    result_2 = create_optimization_history()

    # test a standard call
    visualize.optimizer_history(result_1)

    # test plotting of lists
    visualize.optimizer_history([result_1, result_2])


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
        visualize.optimizer_history(
            result_1,
            y_limits=[-0.5, 2.5],
            start_indices=[0, 1, 4, 11],
            reference=ref_point,
            size=alt_fig_size,
            trace_x="steps",
            trace_y="fval",
            offset_y=-10.0,
            colors=[1.0, 0.3, 0.3, 0.5],
        )

    # Test with linear scale
    visualize.optimizer_history(
        [result_1, result_2],
        y_limits=[-0.5, 2.5],
        start_indices=[0, 1, 4, 11],
        reference=ref_point,
        size=alt_fig_size,
        scale_y="lin",
    )

    # Test with y-limits as float
    visualize.optimizer_history(
        result_1,
        y_limits=5.0,
        start_indices=3,
        reference=ref3,
        trace_x="time",
        trace_y="gradnorm",
    )


@close_fig
def test_optimizer_history_lowlevel():
    # test empty input
    visualize.optimizer_history_lowlevel([])

    # pass numpy array
    x_vals = np.array(list(range(10)))
    y_vals = 11.0 * np.ones(10) - x_vals
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
    """Test pypesto.visualize.optimization_stats"""

    # create the pypesto problem
    problem = create_problem()

    # create optimizer
    optimizer_options = {"maxfun": 200}
    optimizer = optimize.ScipyOptimizer(
        method="TNC", options=optimizer_options
    )

    # run optimization
    result_1 = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=10,
        progress_bar=False,
    )

    result_2 = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=10,
        progress_bar=False,
    )

    visualize.optimization_run_property_per_multistart(
        result_1, "n_fval", legends="best result"
    )

    visualize.optimization_run_property_per_multistart(
        result_1, "n_fval", plot_type="hist", legends="best result"
    )

    visualize.optimization_run_property_per_multistart(result_2, "n_fval")

    # test plotting of lists
    visualize.optimization_run_property_per_multistart(
        [result_1, result_2],
        "n_fval",
        legends=["result1", "result2"],
        plot_type="line",
    )

    visualize.optimization_run_property_per_multistart(
        result_1, "time", plot_type="hist", legends="best result"
    )

    visualize.optimization_run_property_per_multistart(
        [result_1, result_2],
        "time",
        colors=[[0.5, 0.9, 0.9, 0.3], [0.9, 0.7, 0.8, 0.5]],
        legends=["result1", "result2"],
        plot_type="hist",
    )

    visualize.optimization_run_properties_per_multistart([result_1, result_2])

    visualize.optimization_run_properties_one_plot(result_1, ["time"])

    visualize.optimization_run_properties_one_plot(
        result_1, ["n_fval", "n_grad", "n_hess"]
    )

    visualize.optimization_run_property_per_multistart(
        [result_1, result_2],
        "time",
        colors=[[0.5, 0.9, 0.9, 0.3], [0.9, 0.7, 0.8, 0.5]],
        legends=["result1", "result2"],
        plot_type="both",
    )


@close_fig
def test_optimize_convergence():
    result = create_optimization_result()
    result_nan = create_optimization_result_nan_inf()

    visualize.optimizer_convergence(result)
    visualize.optimizer_convergence(result_nan)


def test_assign_clusters():
    # test empty input
    pypesto.util.assign_clusters([])

    # test if it runs at all
    fvals = [0.01, 0.02, 1.01, 2.02, 2.03, 2.04, 3, 4, 4.1, 4.11, 10]
    pypesto.util.assign_clusters(fvals)
    fvals = np.array(fvals)
    clust, clustsize = pypesto.util.assign_clusters(fvals)
    np.testing.assert_array_equal(clust, [0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5])
    np.testing.assert_array_equal(clustsize, [2, 1, 3, 1, 3, 1])

    # test if clustering works as intended
    fvals = [0.0, 0.00001, 1.0, 2.0, 2.001]
    clust, clustsize = pypesto.util.assign_clusters(fvals)
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
    fvals = [0.01, 0.02, 1.0]
    visualize.assign_colors(fvals, colors=[0.5, 0.9, 0.9, 0.3])
    visualize.assign_colors(
        fvals,
        colors=[
            [0.5, 0.9, 0.9, 0.3],
            [0.5, 0.8, 0.8, 0.5],
            [0.9, 0.1, 0.1, 0.1],
        ],
    )


def test_delete_nan_inf():
    # create fvals containing nan and inf
    fvals = np.array([42, 1.5, np.nan, 67.01, np.inf])

    # create a random x
    x = np.array([[1, 2], [1, 1], [np.nan, 1], [65, 1], [2, 3]])
    x, fvals = pypesto.util.delete_nan_inf(fvals, x)

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
    visualize.create_references(references=ref_list_2, x=ref2[0], fval=ref2[1])


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
    n_par = len(result.optimize_result.x[0])
    trace_neglogpost = np.random.randn(n_chain, n_iter)
    trace_neglogprior = np.zeros((n_chain, n_iter))
    trace_x = np.random.randn(n_chain, n_iter, n_par)
    betas = np.array([1, 0.1])
    sample_result = pypesto.McmcPtResult(
        trace_neglogpost=trace_neglogpost,
        trace_neglogprior=trace_neglogprior,
        trace_x=trace_x,
        betas=betas,
        burn_in=10,
    )
    result.sample_result = sample_result

    return result


@close_fig
def test_sampling_fval_traces():
    """Test pypesto.visualize.sampling_fval_traces"""
    result = create_sampling_result()
    visualize.sampling_fval_traces(result)
    # call with custom arguments
    visualize.sampling_fval_traces(
        result, i_chain=1, stepsize=5, size=(10, 10)
    )


@close_fig
def test_sampling_parameter_traces():
    """Test pypesto.visualize.sampling_parameter_traces"""
    result = create_sampling_result()
    visualize.sampling_parameter_traces(result)
    # call with custom arguments
    visualize.sampling_parameter_traces(
        result, i_chain=1, stepsize=5, size=(10, 10), use_problem_bounds=False
    )


@close_fig
def test_sampling_scatter():
    """Test pypesto.visualize.sampling_scatter"""
    result = create_sampling_result()
    visualize.sampling_scatter(result)
    # call with custom arguments
    visualize.sampling_scatter(result, i_chain=1, stepsize=5, size=(10, 10))


@close_fig
def test_sampling_1d_marginals():
    """Test pypesto.visualize.sampling_1d_marginals"""
    result = create_sampling_result()
    visualize.sampling_1d_marginals(result)
    # call with custom arguments
    visualize.sampling_1d_marginals(
        result, i_chain=1, stepsize=5, size=(10, 10)
    )
    # call with other modes
    visualize.sampling_1d_marginals(result, plot_type="hist")
    visualize.sampling_1d_marginals(
        result, plot_type="kde", bw_method="silverman"
    )


@close_fig
def test_sampling_parameter_cis():
    """Test pypesto.visualize.sampling_parameter_cis"""
    result = create_sampling_result()
    visualize.sampling_parameter_cis(result)
    # call with custom arguments
    visualize.sampling_parameter_cis(
        result, alpha=[99, 68], step=0.1, size=(10, 10)
    )


@close_fig
def test_sampling_prediction_trajectories():
    """Test pypesto.visualize.sampling_prediction_trajectories"""
    credibility_interval_levels = [99, 68]
    result = sample_petab_problem()
    post_processor_amici_x = functools.partial(
        post_processor,
        output_type=pypesto.C.AMICI_X,
        output_ids=result.problem.objective.amici_model.get_state_ids(),
    )
    predictor = predict.AmiciPredictor(
        result.problem.objective,
        post_processor=post_processor_amici_x,
        output_ids=result.problem.objective.amici_model.get_state_ids(),
    )

    sample_ensemble = ensemble.Ensemble.from_sample(
        result,
        x_names=result.problem.x_names,
        ensemble_type=pypesto.C.EnsembleType.sample,
        lower_bound=result.problem.lb,
        upper_bound=result.problem.ub,
    )

    ensemble_prediction = sample_ensemble.predict(
        predictor,
        prediction_id=pypesto.C.AMICI_X,
        progress_bar=False,
    )

    # Plot by
    visualize.sampling_prediction_trajectories(
        ensemble_prediction,
        levels=credibility_interval_levels,
        groupby=pypesto.C.CONDITION,
    )
    visualize.sampling_prediction_trajectories(
        ensemble_prediction,
        levels=credibility_interval_levels,
        size=(10, 10),
        groupby=pypesto.C.OUTPUT,
    )


@close_fig
def test_visualize_optimized_model_fit():
    """Test pypesto.visualize.visualize_optimized_model_fit"""
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(
        os.path.join(current_path, "..", "..", "doc", "example")
    )

    # import to petab
    petab_problem = petab.Problem.from_yaml(
        os.path.join(
            dir_path, "conversion_reaction", "conversion_reaction.yaml"
        )
    )
    # import to pypesto
    importer = pypesto.petab.PetabImporter(petab_problem)
    # create problem
    problem = importer.create_problem()

    result = optimize.minimize(
        problem=problem,
        n_starts=1,
        progress_bar=False,
    )

    # test call of visualize_optimized_model_fit
    visualize_optimized_model_fit(
        petab_problem=petab_problem,
        result=result,
        pypesto_problem=problem,
    )


@close_fig
def test_visualize_optimized_model_fit_aggregated():
    """Test pypesto.visualize.visualize_optimized_model_fit with an AggregatedObjective"""
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(
        os.path.join(current_path, "..", "..", "doc", "example")
    )

    # import to petab
    petab_problem = petab.Problem.from_yaml(
        os.path.join(
            dir_path, "conversion_reaction", "conversion_reaction.yaml"
        )
    )

    # add prior to the problem
    petab_problem.parameter_df["objectivePriorType"] = "uniform"
    objectivePriorParameters = [
        f"{lb};{ub}"
        for lb, ub in zip(
            petab_problem.parameter_df.lowerBound,
            petab_problem.parameter_df.upperBound,
        )
    ]
    petab_problem.parameter_df["objectivePriorParameters"] = (
        objectivePriorParameters
    )

    # import to pypesto
    importer = pypesto.petab.PetabImporter(petab_problem)
    # create problem
    problem = importer.create_problem()
    # check if the objective is an AggregatedObjective
    assert isinstance(problem.objective, pypesto.objective.AggregatedObjective)

    result = optimize.minimize(
        problem=problem,
        n_starts=1,
        progress_bar=False,
    )

    # test call of visualize_optimized_model_fit
    visualize_optimized_model_fit(
        petab_problem=petab_problem,
        result=result,
        pypesto_problem=problem,
    )


@close_fig
def test_time_trajectory_model():
    """Test pypesto.visualize.time_trajectory_model"""
    current_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(
        os.path.join(current_path, "..", "..", "doc", "example")
    )

    # import to petab
    petab_problem = petab.Problem.from_yaml(
        os.path.join(
            dir_path, "conversion_reaction", "conversion_reaction.yaml"
        )
    )
    # import to pypesto
    importer = pypesto.petab.PetabImporter(petab_problem)
    # create problem
    problem = importer.create_problem()

    result = optimize.minimize(
        problem=problem,
        n_starts=1,
        progress_bar=False,
    )

    # test call of time_trajectory_model
    time_trajectory_model(result=result)


def test_monotonic_history():
    from pypesto.history.memory import MemoryHistory
    from pypesto.visualize.optimizer_history import monotonic_history

    def create_history(t, fx):
        from pypesto.C import FVAL, TIME

        history = MemoryHistory()
        history._trace[TIME] = t
        history._trace[FVAL] = fx
        assert (history.get_time_trace() == t).all()
        assert (history.get_fval_trace() == fx).all()
        return history

    t = np.arange(5, dtype=float)
    history1 = create_history(t, -t)
    t_mono, fx_mono = monotonic_history([history1, history1])
    assert t_mono.tolist() == history1.get_time_trace()
    assert fx_mono.tolist() == history1.get_fval_trace()

    history2 = create_history(t, -2 * t)
    for histories in (
        [history1, history2],
        [history2, history1],
        [history1, history2, history1],
        [history2, history1, history2],
    ):
        t_mono, fx_mono = monotonic_history(histories)
        assert t_mono.tolist() == history2.get_time_trace()
        assert fx_mono.tolist() == history2.get_fval_trace()

    t = np.arange(0.5, 6.5, dtype=float)
    history3 = create_history(t, 2 - 2 * t)
    for histories in (
        [history1, history3],
        [history3, history1],
        [history1, history3, history1],
    ):
        t_mono, fx_mono = monotonic_history(histories)
        assert t_mono.tolist() == [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5]
        assert fx_mono.tolist() == [
            -0.0,
            -1.0,
            -1.0,
            -2.0,
            -3.0,
            -3.0,
            -5.0,
            -7.0,
            -9.0,
        ]


@close_fig
def test_sacess_history():
    """Test pypesto.visualize.optimizer_history.sacess_history"""
    from pypesto.optimize.ess.sacess import ESSExitFlag, SacessOptimizer
    from pypesto.visualize.optimizer_history import sacess_history

    problem = create_problem()
    sacess = SacessOptimizer(
        max_walltime_s=1, num_workers=2, sacess_loglevel=logging.WARNING
    )
    sacess.minimize(problem)
    assert sacess.exit_flag == ESSExitFlag.MAX_TIME
    sacess_history(sacess.histories)


@pytest.mark.parametrize(
    "result_creation",
    [create_optimization_result, create_optimization_result_nan_inf],
)
@close_fig
def test_parameters_correlation_matrix(result_creation):
    """Test pypesto.visualize.parameters_correlation_matrix"""
    result = result_creation()

    visualize.parameters_correlation_matrix(result)


@close_fig
def test_plot_ordinal_categories():
    example_ordinal_yaml = (
        Path(__file__).parent
        / ".."
        / ".."
        / "doc"
        / "example"
        / "example_ordinal"
        / "example_ordinal.yaml"
    )
    petab_problem = petab.Problem.from_yaml(example_ordinal_yaml)
    # Set seed for reproducibility.
    np.random.seed(0)
    optimizer = pypesto.optimize.ScipyOptimizer(
        method="L-BFGS-B", options={"maxiter": 1}
    )
    importer = pypesto.petab.PetabImporter(petab_problem, hierarchical=True)
    problem = importer.create_problem()
    result = pypesto.optimize.minimize(
        problem=problem, n_starts=1, optimizer=optimizer
    )
    visualize.plot_categories_from_pypesto_result(result)


@close_fig
def test_visualize_estimated_observable_mapping():
    example_semiquantitative_yaml = (
        Path(__file__).parent
        / ".."
        / ".."
        / "doc"
        / "example"
        / "example_semiquantitative"
        / "example_semiquantitative_linear.yaml"
    )
    petab_problem = petab.Problem.from_yaml(example_semiquantitative_yaml)
    # Set seed for reproducibility.
    np.random.seed(0)
    optimizer = pypesto.optimize.ScipyOptimizer(
        method="L-BFGS-B",
        options={
            "disp": None,
            "ftol": 2.220446049250313e-09,
            "gtol": 1e-5,
            "maxiter": 1,
        },
    )
    importer = pypesto.petab.PetabImporter(petab_problem, hierarchical=True)
    problem = importer.create_problem()
    result = pypesto.optimize.minimize(
        problem=problem, n_starts=1, optimizer=optimizer
    )
    visualize.visualize_estimated_observable_mapping(result, problem)
