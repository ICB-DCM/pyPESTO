"""Pointwise observation predictions for explicit additive Gaussian noise.

Prototype for issue #1699. The empirical ensemble is a finite mixture, not
an estimate of a continuous posterior beyond the supplied members.
"""

from math import fsum

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri

from ..C import MEAN, MEDIAN, STANDARD_DEVIATION
from ..result import PredictionConditionResult, PredictionResult


def _normalized_weights(weights):
    """Avoid unnecessary rescaling, but safely handle an overflowing total."""
    try:
        total = fsum(weights)
    except OverflowError:
        weights = weights / np.max(weights)
        total = fsum(weights)
    return weights / total


def normal_mixture_quantiles(means, sigmas, probabilities, weights=None):
    """Return generalized-inverse CDF quantiles along the last (member) axis.

    ``sigmas`` must exactly match ``means``; zero sigma denotes a point mass.
    ``weights`` are nonnegative linear mixture masses, NOT log likelihoods.
    With no noise this is an empirical inverse CDF (no linear interpolation),
    so it deliberately differs from NumPy's default percentile convention.
    The returned shape is ``(n_probabilities, *means.shape[:-1])``.
    Finite inputs only; probabilities include endpoints. Normal components
    have infinite endpoint quantiles. This does not specify joint dependence
    across time or observables and cannot produce simultaneous bands.

    Parameters
    ----------
    means : array_like
        Finite component means. The last axis contains the ensemble members.
    sigmas : array_like
        Finite, nonnegative component standard deviations, with exactly the
        same shape as ``means``. Zero denotes a point mass at the mean.
    probabilities : array_like
        One-dimensional sequence of finite probabilities in ``[0, 1]``.
    weights : array_like or None
        One nonnegative finite linear mass per member, with positive total.
        Masses are normalized internally; ``None`` gives equal weights.

    Returns
    -------
    numpy.ndarray
        Quantiles with shape ``(len(probabilities), *means.shape[:-1])``.

    Raises
    ------
    ValueError
        If inputs are invalid or component quantiles exceed floating-point
        range.
    ArithmeticError
        If normal-tail comparisons exceed floating-point precision or
        bisection does not converge.
    """
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if means.ndim < 1 or means.shape[-1] == 0 or sigmas.shape != means.shape:
        raise ValueError(
            "means and sigmas must match, with a nonempty member axis"
        )
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(sigmas)):
        raise ValueError("means and sigmas must be finite")
    if np.any(sigmas < 0):
        raise ValueError("sigmas must be nonnegative")
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be a finite vector in [0, 1]")
    w = (
        np.ones(means.shape[-1])
        if weights is None
        else np.asarray(weights, float)
    )
    if (
        w.shape != (means.shape[-1],)
        or not np.all(np.isfinite(w))
        or np.any(w < 0)
        or not np.any(w > 0)
    ):
        raise ValueError(
            "weights must be finite nonnegative masses with positive total"
        )
    active = w > 0
    w = _normalized_weights(w[active])
    m = means[..., active].reshape(-1, active.sum())
    s = sigmas[..., active].reshape(m.shape)
    answer = np.empty((len(p), len(m)))
    for j, (mu, sigma) in enumerate(zip(m, s, strict=True)):
        positive = sigma > 0
        atoms = np.unique(mu[~positive])

        def cdf_sign(
            x, probability, left=False, mu=mu, sigma=sigma, positive=positive
        ):
            # Compare F(x)-p without rounding a near-one normal CDF to one.
            # Integer/atom mass is separated from small normal tail terms.
            with np.errstate(over="ignore"):
                z = (x - mu[positive]) / sigma[positive]
            normal_weights = w[positive]
            upper = z >= 0
            atom_mask = (mu[~positive] < x) if left else (mu[~positive] <= x)
            baseline = fsum(
                [
                    *normal_weights[upper],
                    *w[~positive][atom_mask],
                    -float(probability),
                ]
            )
            positive_logs = np.log(normal_weights[~upper]) + log_ndtr(
                z[~upper]
            )
            negative_logs = np.log(normal_weights[upper]) + log_ndtr(-z[upper])
            if baseline > 0:
                positive_logs = np.append(positive_logs, np.log(baseline))
            elif baseline < 0:
                negative_logs = np.append(negative_logs, np.log(-baseline))
            log_positive, log_negative = (
                logsumexp(positive_logs),
                logsumexp(negative_logs),
            )
            if log_positive == log_negative == -np.inf and np.any(positive):
                raise ArithmeticError(
                    "normal tail balance exceeds floating-point precision"
                )
            return int(log_positive > log_negative) - int(
                log_positive < log_negative
            )

        for i, probability in enumerate(p):
            if probability in (0, 1):
                answer[i, j] = (
                    (-np.inf if probability == 0 else np.inf)
                    if np.any(positive)
                    else (mu.min() if probability == 0 else mu.max())
                )
                continue
            if not np.any(positive):
                order = np.argsort(mu)
                index = np.searchsorted(
                    np.cumsum(w[order]), probability, side="left"
                )
                answer[i, j] = mu[order[min(index, len(order) - 1)]]
                continue
            hit = next(
                (
                    a
                    for a in atoms
                    if cdf_sign(a, probability, left=True)
                    <= 0
                    <= cdf_sign(a, probability)
                ),
                None,
            )
            if hit is not None:
                answer[i, j] = hit
                continue
            component_quantiles = mu + sigma * ndtri(probability)
            lo, hi = component_quantiles.min(), component_quantiles.max()
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(
                    "component quantiles exceed floating-point range"
                )
            # Cover the full float64 exponent range, not a fixed accuracy
            # tied to the initial separation between mixture components.
            for _ in range(2200):
                mid = lo / 2 + hi / 2
                if mid == lo or mid == hi:
                    break
                sign = cdf_sign(mid, probability)
                if sign == 0:
                    hi = mid
                    break
                if sign > 0:
                    hi = mid
                else:
                    lo = mid
            else:
                raise ArithmeticError("mixture quantile failed to converge")
            answer[i, j] = hi
    return answer.reshape((len(p), *means.shape[:-1]))


def compute_noisy_summary(
    ensemble_prediction, percentiles=(5, 50, 95), weights=None
):
    """Summarize an EnsemblePrediction with explicit Gaussian output_sigmay.

    Return a separate dictionary of PredictionResult objects; leave existing
    prediction_summary, outputs, sensitivities and caches untouched. Only
    additive normal noise in original output units is supported. Caller must
    ensure output_sigmay refers to these outputs, not transformed variables.
    All members must have matching condition/output IDs, times and 2D shapes.
    No noise is inferred from an AMICI model or missing sigma values.
    Sensitivities of predictive summaries are not provided.

    Parameters
    ----------
    ensemble_prediction : EnsemblePrediction
        Nonempty ensemble whose conditions contain finite ``output`` and
        finite, nonnegative ``output_sigmay`` arrays of identical shape
        ``(timepoints, outputs)``. Standard deviations may vary by member,
        condition, time and output.
    percentiles : sequence of int or float
        Requested pointwise percentile levels in ``[0, 100]``. Labels follow
        ``get_percentile_label``, including each level's int/float type.
        Distinct levels must not round to the same dictionary label.
    weights : array_like or None
        Nonnegative finite linear masses with positive total, one per member.
        ``None`` gives equal weights. Log likelihoods are not accepted as
        weights and ``output_weight`` is not used automatically.

    Returns
    -------
    dict[str, PredictionResult]
        New results containing the observation mean, standard deviation,
        median and requested percentiles. Each condition preserves the input
        timepoints and output IDs; summary sensitivities are absent.

    Raises
    ------
    ValueError
        If predictions are empty or misaligned, output/sigma arrays or weights
        are invalid, percentile labels collide, or moments or component
        quantiles exceed floating-point range.
    ArithmeticError
        If the mixture quantile calculation cannot resolve a normal-tail
        comparison or does not converge.
    """
    from .ensemble import get_percentile_label

    predictions = ensemble_prediction.prediction_results
    if not predictions or not predictions[0].conditions:
        raise ValueError("nonempty prediction results and conditions required")
    percs = np.asarray(percentiles, float)
    # Reuse strict numerical validation, including all weights and probabilities.
    probabilities = np.concatenate(([0.5], percs / 100))
    first = predictions[0]
    keys = [MEAN, STANDARD_DEVIATION, MEDIAN] + [
        get_percentile_label(p) for p in percentiles
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("percentile labels must be distinct after rounding")
    result = {key: [] for key in keys}
    for prediction in predictions:
        if list(prediction.condition_ids) != list(first.condition_ids) or len(
            prediction.conditions
        ) != len(first.conditions):
            raise ValueError("condition IDs/order must match between members")
    for ic, reference in enumerate(first.conditions):
        means, sigmas = [], []
        for prediction in predictions:
            cond = prediction.conditions[ic]
            if list(cond.output_ids) != list(
                reference.output_ids
            ) or not np.array_equal(cond.timepoints, reference.timepoints):
                raise ValueError("output IDs/order and timepoints must match")
            if cond.output is None or cond.output_sigmay is None:
                raise ValueError("each output requires explicit output_sigmay")
            mu, sigma = (
                np.asarray(cond.output, float),
                np.asarray(cond.output_sigmay, float),
            )
            if (
                mu.shape != (len(cond.timepoints), len(cond.output_ids))
                or sigma.shape != mu.shape
            ):
                raise ValueError(
                    "output and sigma must have shape (timepoints, outputs)"
                )
            means.append(mu)
            sigmas.append(sigma)
        means, sigmas = np.stack(means, axis=-1), np.stack(sigmas, axis=-1)
        quantiles = normal_mixture_quantiles(
            means, sigmas, probabilities, weights
        )
        w = (
            np.ones(len(predictions))
            if weights is None
            else np.asarray(weights, float)
        )
        w = _normalized_weights(w)
        mean = np.sum(means * w, axis=-1)
        std = np.sqrt(
            np.sum(w * ((means - mean[..., None]) ** 2 + sigmas**2), axis=-1)
        )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("summary moments exceed floating-point range")
        values = {MEAN: mean, STANDARD_DEVIATION: std, MEDIAN: quantiles[0]}
        values.update(
            {
                get_percentile_label(p): q
                for p, q in zip(percentiles, quantiles[1:], strict=True)
            }
        )
        for key in result:
            result[key].append(
                PredictionConditionResult(
                    timepoints=np.array(reference.timepoints, copy=True),
                    output_ids=list(reference.output_ids),
                    output=values[key],
                )
            )
    return {
        key: PredictionResult(
            conditions=value,
            condition_ids=list(first.condition_ids),
            comment="Gaussian observation predictive " + key,
        )
        for key, value in result.items()
    }
