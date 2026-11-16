"""Checks against analytic distributions, invariants, and binomial sampling."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import ndtr, ndtri

from pypesto.C import MEAN, STANDARD_DEVIATION
from pypesto.ensemble import (
    EnsemblePrediction,
    compute_noisy_summary,
    get_percentile_label,
)
from pypesto.ensemble import normal_mixture_quantiles as quantiles
from pypesto.result import PredictionConditionResult, PredictionResult


def ensemble(sigmas=(0.3, 0.8)):
    return EnsemblePrediction(
        prediction_results=[
            PredictionResult(
                conditions=[
                    PredictionConditionResult(
                        timepoints=np.array([0.0, 1.0]),
                        output_ids=["a"],
                        output=np.array([[m], [m + 1]]),
                        output_sigmay=np.full((2, 1), s),
                    )
                ],
                condition_ids=["c"],
            )
            for m, s in zip([0.0, 2.0], sigmas, strict=True)
        ]
    )


@pytest.mark.parametrize("mu,sigma", [(0, 1), (17, 0.001), (-3, 20)])
def test_single_normal(mu, sigma):
    p = np.array([1e-8, 0.025, 0.5, 0.975, 1 - 1e-8])
    assert_allclose(
        quantiles([mu], [sigma], p),
        mu + sigma * ndtri(p),
        atol=1e-12,
        rtol=1e-12,
    )


def test_cdf_and_invariances():
    m, s, w = (
        np.array([-4.0, 1.0, 5.0]),
        np.array([0.2, 2.0, 0.7]),
        np.array([1.0, 4.0, 2.0]),
    )
    p = np.array([0.001, 0.025, 0.2, 0.5, 0.9, 0.999])
    q = quantiles(m, s, p, w)
    assert_allclose(ndtr((q[:, None] - m) / s) @ (w / w.sum()), p, atol=2e-14)
    assert_allclose(quantiles(m[::-1], s[::-1], p, w[::-1]), q, atol=1e-13)
    assert_allclose(quantiles(m, s, p, w * 1e300), q, atol=1e-13)
    assert_allclose(quantiles(m * 3 + 9, s * 3, p, w), q * 3 + 9, atol=1e-12)
    assert_allclose(
        quantiles(np.r_[m, m], np.r_[s, s], p, np.r_[w, w]), q, atol=1e-13
    )


def test_zero_noise_empirical_inverse_and_atoms():
    assert_array_equal(
        quantiles(
            [0, 2, 9], [0, 0, 0], [0, 0.2, 0.21, 0.5, 1], [0.2, 0.3, 0.5]
        ),
        [0, 0, 2, 2, 9],
    )
    # Half standard normal and half point mass at zero: CDF jump .25 -> .75.
    assert_array_equal(quantiles([0, 0], [1, 0], [0.3, 0.5, 0.7]), [0, 0, 0])
    assert_allclose(
        quantiles([0, 0], [1, 0], [0.1, 0.9]),
        [ndtri(0.2), ndtri(0.8)],
        atol=1e-13,
    )
    assert_array_equal(quantiles([0, 2], [1, 0], [0, 1]), [-np.inf, np.inf])
    assert_allclose(quantiles([1, 1000], [2, 500], [0.5], [1, 0]), [1])


@pytest.mark.parametrize(
    "m,s,p,w",
    [
        ([], [], [0.5], None),
        ([0], [-1], [0.5], None),
        ([np.nan], [1], [0.5], None),
        ([0], [np.inf], [0.5], None),
        ([0], [1], [-0.1], None),
        ([0], [1], [1.1], None),
        ([0], [1], [np.nan], None),
        ([0], [1], [[0.5]], None),
        ([0], [1], [0.5], [-1]),
        ([0], [1], [0.5], [0]),
        ([0], [1], [0.5], [np.inf]),
        ([0], [1], [0.5], [1, 2]),
        ([0], [1, 2], [0.5], None),
    ],
)
def test_invalid(m, s, p, w):
    with pytest.raises(ValueError):
        quantiles(m, s, p, w)


def test_integration_moments_metadata_and_no_mutation():
    ens = ensemble()
    original = deepcopy(ens.prediction_results)
    latent = ens.compute_summary()
    result = compute_noisy_summary(ens, weights=[0.25, 0.75])
    assert_allclose(result[MEAN].conditions[0].output[:, 0], [1.5, 2.5])
    # E[var(Y|theta)] + var(E[Y|theta])
    assert_allclose(
        result[STANDARD_DEVIATION].conditions[0].output,
        np.sqrt(0.25 * 0.3**2 + 0.75 * 0.8**2 + 0.75),
    )
    assert result[MEAN].condition_ids == ["c"]
    assert result[MEAN].conditions[0].output_ids == ["a"]
    assert result[MEAN].conditions[0].output_sensi is None
    assert ens.prediction_summary is latent
    for before, after in zip(original, ens.prediction_results, strict=True):
        assert_array_equal(
            before.conditions[0].output, after.conditions[0].output
        )
        assert_array_equal(
            before.conditions[0].output_sigmay,
            after.conditions[0].output_sigmay,
        )
    result[MEAN].conditions[0].timepoints[0] = 500
    assert ens.prediction_results[0].conditions[0].timepoints[0] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "missing_sigma",
        "negative_sigma",
        "shape",
        "time",
        "output_id",
        "condition_id",
        "empty",
    ],
)
def test_integration_validation(fault):
    ens = ensemble()
    cond = ens.prediction_results[1].conditions[0]
    if fault == "missing_sigma":
        cond.output_sigmay = None
    if fault == "negative_sigma":
        cond.output_sigmay[0, 0] = -1
    if fault == "shape":
        cond.output_sigmay = np.ones(1)
    if fault == "time":
        cond.timepoints[0] = 2
    if fault == "output_id":
        cond.output_ids = ["wrong"]
    if fault == "condition_id":
        ens.prediction_results[1].condition_ids = ["wrong"]
    if fault == "empty":
        ens.prediction_results = []
    with pytest.raises(ValueError):
        compute_noisy_summary(ens)


def test_multicondition_shapes():
    ens = ensemble()
    for pred in ens.prediction_results:
        pred.condition_ids.append("d")
        pred.conditions.append(
            PredictionConditionResult(
                timepoints=[0],
                output_ids=["x", "y"],
                output=np.array([[4.0, 5.0]]),
                output_sigmay=np.array([[0.1, 0.2]]),
            )
        )
    result = compute_noisy_summary(ens)
    assert result[MEAN].conditions[1].output.shape == (1, 2)
    assert_allclose(
        result[get_percentile_label(5)].conditions[1].output,
        [[4 + 0.1 * ndtri(0.05), 5 + 0.2 * ndtri(0.05)]],
    )


def test_independent_monte_carlo():
    rng = np.random.default_rng(1699)
    n = 400_000
    m, s, w = (
        np.array([-2.0, 1.0, 4.0]),
        np.array([0.5, 1.5, 0.2]),
        np.array([0.2, 0.5, 0.3]),
    )
    p = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    q = quantiles(m, s, p, w)
    component = rng.choice(3, size=n, p=w)
    samples = rng.normal(m[component], s[component])
    empirical = np.mean(samples[:, None] <= q, axis=0)
    # Fixed before observing outcome: six binomial standard errors + 1/n.
    assert np.all(np.abs(empirical - p) < 6 * np.sqrt(p * (1 - p) / n) + 1 / n)


def test_zero_noise_summary():
    ens = ensemble(sigmas=(0, 0))
    result = compute_noisy_summary(ens)
    assert_array_equal(
        result[get_percentile_label(5)].conditions[0].output[:, 0], [0, 1]
    )
    assert_array_equal(
        result[get_percentile_label(95)].conditions[0].output[:, 0], [2, 3]
    )
    assert_allclose(result[STANDARD_DEVIATION].conditions[0].output, 1)


@pytest.mark.parametrize("separation", [1e40, 1e200])
def test_wide_bracket_converges(separation):
    # The far component contributes negligible CDF near the mixture median.
    expected = ndtri(0.5 / 0.9)
    got = quantiles([0, separation], [1, 1], [0.5], [0.9, 0.1])
    assert_allclose(got, [expected], rtol=1e-12, atol=1e-12)
    assert_allclose(
        0.9 * ndtr(got) + 0.1 * ndtr(got - separation), 0.5, atol=1e-14
    )


@pytest.mark.parametrize("percentiles", [[2.501, 2.504], [5, 5]])
def test_percentile_label_collision_rejected(percentiles):
    with pytest.raises(ValueError, match="labels"):
        compute_noisy_summary(ensemble(), percentiles=percentiles)


@pytest.mark.parametrize("with_atom", [False, True])
def test_balanced_normal_tails(with_atom):
    means, sigmas, weights, p = [0, 100], [1, 1], [0.5, 0.5], 0.5
    if with_atom:
        means, sigmas, weights, p = (
            [0, 100, 90],
            [1, 1, 0],
            [0.25, 0.25, 0.5],
            0.25,
        )
    assert_allclose(
        quantiles(means, sigmas, [p], weights), [50], atol=1e-12, rtol=0
    )


def test_unrepresentable_tail_balance_rejected():
    with pytest.raises(ArithmeticError, match="precision"):
        quantiles([0, 1e200], [1, 1], [0.5])


def test_weight_normalization_at_component_boundary():
    a = quantiles([0, 100], [1, 1], [0.1], [1, 9])
    b = quantiles([0, 100], [1, 1], [0.1], [0.1, 0.9])
    assert_allclose(a, b, rtol=0, atol=1e-12)
