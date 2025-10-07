import os
from functools import partial

import numpy as np
import scipy.optimize as so

import pypesto
import pypesto.optimize as optimize
import pypesto.sample as sample
from pypesto.C import (
    AMICI_STATUS,
    AMICI_T,
    AMICI_Y,
    MEAN,
    POINTWISE,
    WEIGHTED_SIGMA,
)
from pypesto.engine import MultiProcessEngine
from pypesto.ensemble import (
    Ensemble,
    calculate_cutoff,
    read_ensemble_prediction_from_h5,
    write_ensemble_prediction_to_h5,
)
from pypesto.predict import AmiciPredictor

from ..visualize import create_petab_problem


def test_ensemble_from_optimization():
    """
    Test reading an ensemble from optimization result.
    """
    objective = pypesto.Objective(
        fun=so.rosen, grad=so.rosen_der, hess=so.rosen_hess
    )
    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))
    n_starts = 5

    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    optimizer = optimize.ScipyOptimizer(options={"maxiter": 10})
    history_options = pypesto.HistoryOptions(trace_record=True)
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=n_starts,
        history_options=history_options,
        progress_bar=False,
    )

    # change fvals of each start
    for i_start, optimizer_result in enumerate(result.optimize_result.list):
        optimizer_result["fval"] = i_start
        for i_iter in range(len(optimizer_result["history"]._trace["fval"])):
            optimizer_result["history"]._trace["fval"][i_iter] = (
                len(optimizer_result["history"]._trace["fval"])
                + i_start
                - i_iter
            )

    # test_endpoints
    ensemble_ep = Ensemble.from_optimization_endpoints(
        result=result, rel_cutoff=3, max_size=10
    )

    ensemble_hist = Ensemble.from_optimization_history(
        result=result, rel_cutoff=3, max_size=10, max_per_start=5
    )

    # compare vector_tags with the expected values:
    ep_tags = [
        (result.optimize_result.list[i]["id"], -1) for i in [0, 1, 2, 3]
    ]

    hist_tags = [
        (
            result.optimize_result.list[i]["id"],
            len(result.optimize_result.list[i]["history"]._trace["fval"])
            - 1
            - j,
        )
        for i in range(3)
        for j in reversed(range(3 - i))
    ]
    assert hist_tags == ensemble_hist.vector_tags
    assert ep_tags == ensemble_ep.vector_tags


def test_cutoff_computation():
    """
    Test computing ensemble cutoff based on chi^2 distribution.
    """
    from scipy.stats import chi2

    objective = pypesto.Objective(
        fun=so.rosen, grad=so.rosen_der, hess=so.rosen_hess
    )
    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))
    n_starts = 5

    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    optimizer = optimize.ScipyOptimizer(options={"maxiter": 10})
    history_options = pypesto.HistoryOptions(trace_record=True)
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=n_starts,
        history_options=history_options,
        progress_bar=False,
    )

    assert calculate_cutoff(result, percentile=95) == (
        result.optimize_result.fval[0]
        + chi2.ppf(q=0.95, df=result.problem.dim) / 2
    )

    ens = Ensemble.from_optimization_endpoints(result=result, percentile=95)
    assert ens.n_vectors == sum(
        result.optimize_result.fval <= calculate_cutoff(result, percentile=95)
    )

    assert calculate_cutoff(result, percentile=95, cr_option=POINTWISE) == (
        result.optimize_result.fval[0] + chi2.ppf(q=0.95, df=1) / 2
    )


def test_parameter_ci_computation_from_ensemble():
    """
    Test computing ensemble cutoff based on chi^2 distribution and resulting parameter CIs.
    """
    from scipy.stats import norm

    # Normal distribution parameters
    loc = 3
    scale = 1.4

    # Confidence interval
    percentile = 95

    # Upper bound of PPF
    lower_bound = (100 - percentile) / 100 / 2
    upper_bound = 1 - lower_bound

    # The ground truth confidence interval bound will be deviated by this amount to generate
    # the samples.
    # This should be small -- an incorrect chi2 cutoff computation will choose one of the
    # deviations from the bound positions, rather than the bound position, as the bounds of the
    # confidence interval.
    epsilon = 1e-3

    samples = [
        # the lower bound +- epsilon
        norm.ppf(lower_bound, loc=loc, scale=scale) - epsilon,
        norm.ppf(lower_bound, loc=loc, scale=scale),
        norm.ppf(lower_bound, loc=loc, scale=scale) + epsilon,
        # MLE
        loc,
        # the upper bound +- epsilon
        norm.ppf(upper_bound, loc=loc, scale=scale) - epsilon,
        norm.ppf(upper_bound, loc=loc, scale=scale),
        norm.ppf(upper_bound, loc=loc, scale=scale) + epsilon,
    ]
    fvals = [-norm.logpdf(s, loc=loc, scale=scale) for s in samples]
    pairs = list(zip(samples, fvals, strict=True))
    pairs.sort(key=lambda pair: pair[1])

    result = pypesto.Result(
        problem=pypesto.Problem(
            objective=pypesto.Objective(),
            lb=[loc - 3 * scale],
            ub=[loc + 3 * scale],
            dim_full=1,
        )
    )
    for k, (s, fval) in enumerate(pairs):
        result.optimize_result.append(
            optimize_result=pypesto.OptimizerResult(id=str(k), fval=fval, x=s)
        )

    cutoff = calculate_cutoff(result, percentile=percentile)
    ensemble = [
        r for r in result.optimize_result.list if r.fval - 1.0e-14 <= cutoff
    ]  # correction for numerical noise
    xs = [r.x for r in ensemble]
    assert min(xs) == norm.ppf(lower_bound, loc=loc, scale=scale)
    assert max(xs) == norm.ppf(upper_bound, loc=loc, scale=scale)


def test_ensemble_prediction_from_hdf5():
    """
    Test writing an ensemble prediction to hdf5 and loading it.
    """
    problem = create_petab_problem()

    def post_processor(amici_outputs, output_type, output_ids):
        outputs = [
            (
                amici_output[output_type]
                if amici_output[AMICI_STATUS] == 0
                else np.full(
                    (len(amici_output[AMICI_T]), len(output_ids)), np.nan
                )
            )
            for amici_output in amici_outputs
        ]
        return outputs

    post_processor_y = partial(
        post_processor,
        output_type=AMICI_Y,
        output_ids=problem.objective.amici_model.get_observable_ids(),
    )
    predictor_y = AmiciPredictor(
        problem.objective,
        post_processor=post_processor_y,
        output_ids=problem.objective.amici_model.get_observable_ids(),
    )
    ensemble_prediction = get_ensemble_prediction(max_size=10)

    fn = "test_file.hdf5"
    try:
        write_ensemble_prediction_to_h5(ensemble_prediction, fn)
        ensemble_prediction_r = read_ensemble_prediction_from_h5(
            predictor_y, fn
        )

        # test both Ensemble.Predictions
        assert (
            ensemble_prediction.prediction_id
            == ensemble_prediction_r.prediction_id
        )
        for i_run, _ in enumerate(ensemble_prediction.prediction_results):
            assert (
                ensemble_prediction.prediction_results[i_run]
                == ensemble_prediction_r.prediction_results[i_run]
            )
    finally:
        if os.path.exists(fn):
            os.remove(fn)


def test_ensemble_weighted_trajectory():
    """
    Test computing a weighted mean trajectory
    and weighted sigmas for an ensemble.
    """
    ens_pred = get_ensemble_prediction(inc_weights=True, inc_sigmay=True)

    # artifically set the values
    ens_pred.prediction_results[0].conditions[0].output = np.ones((10, 1))
    ens_pred.prediction_results[1].conditions[0].output = 2 * np.ones((10, 1))
    ens_pred.prediction_results[0].conditions[0].output_weight = np.log(10)
    ens_pred.prediction_results[1].conditions[0].output_weight = np.log(20)
    ens_pred.prediction_results[0].conditions[0].output_sigmay = 0.5 * np.ones(
        (10, 1)
    )
    ens_pred.prediction_results[1].conditions[0].output_sigmay = 0.1 * np.ones(
        (10, 1)
    )

    ens_pred.compute_summary(weighting=True, compute_weighted_sigma=True)

    np.testing.assert_almost_equal(
        ens_pred.prediction_summary[MEAN].conditions[0].output,
        5 / 3 * np.ones((10, 1)),
        decimal=15,
    )
    np.testing.assert_almost_equal(
        ens_pred.prediction_summary[WEIGHTED_SIGMA].conditions[0].output,
        0.3 * np.ones((10, 1)),
        decimal=15,
    )


def get_ensemble_prediction(
    max_size: int = 2, inc_weights: bool = False, inc_sigmay: bool = False
):
    """
    Creates an ensemble prediction for the tests.
    """
    problem = create_petab_problem()

    optimizer = optimize.ScipyOptimizer()
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=2,
        progress_bar=False,
    )

    ensemble_ep = Ensemble.from_optimization_endpoints(
        result=result, max_size=10
    )

    # This post_processor will transform the output of the simulation tool
    # such that the output is compatible with the next steps.
    def post_processor(amici_outputs, output_type, output_ids):
        outputs = [
            (
                amici_output[output_type]
                if amici_output[AMICI_STATUS] == 0
                else np.full(
                    (len(amici_output[AMICI_T]), len(output_ids)), np.nan
                )
            )
            for amici_output in amici_outputs
        ]
        return outputs

    amici_objective = result.problem.objective
    observable_ids = amici_objective.amici_model.get_observable_ids()
    post_processor_y = partial(
        post_processor,
        output_type=AMICI_Y,
        output_ids=observable_ids,
    )
    # Create pyPESTO predictors for states and observables
    predictor_y = AmiciPredictor(
        amici_objective,
        post_processor=post_processor_y,
        output_ids=observable_ids,
    )
    engine = MultiProcessEngine()
    ensemble_prediction = ensemble_ep.predict(
        predictor_y,
        prediction_id=AMICI_Y,
        engine=engine,
        include_llh_weights=inc_weights,
        include_sigmay=inc_sigmay,
        progress_bar=False,
    )
    return ensemble_prediction


def test_hpd_calculation():
    """Test the calculation of Highest Posterior Density (HPD)."""
    problem = create_petab_problem()

    sampler = sample.AdaptiveMetropolisSampler(
        options={"show_progress": False}
    )

    result = optimize.minimize(
        problem=problem,
        n_starts=3,
        progress_bar=False,
    )

    result = sample.sample(
        problem=problem,
        sampler=sampler,
        n_samples=100,
        result=result,
    )

    # Manually set up sample (only for testing)
    burn_in = 1
    result.sample_result.burn_in = burn_in
    result.sample_result.trace_neglogpost[0][1:] = np.random.permutation(
        np.arange(len(result.sample_result.trace_neglogpost[0][1:]))
    )

    hpd_ensemble = Ensemble.from_sample(
        result=result, remove_burn_in=True, ci_level=0.95
    )

    expected_length = (
        int((result.sample_result.trace_x[0][burn_in:].shape[0]) * 0.95) + 1
    )
    # Check that the HPD parameters have the expected shape
    assert hpd_ensemble.x_vectors.shape == (problem.dim, expected_length)
    x_indices = np.where(result.sample_result.trace_neglogpost[0][1:] <= 95)[0]
    assert np.all(
        [
            np.any(np.all(x[:, None] == hpd_ensemble.x_vectors, axis=0))
            for x in result.sample_result.trace_x[0][burn_in:][x_indices]
        ]
    )
