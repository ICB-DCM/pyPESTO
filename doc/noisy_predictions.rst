Pointwise predictions with Gaussian measurement noise
============================================================

``compute_noisy_summary`` combines an empirical ensemble of model predictions
with explicitly supplied additive Gaussian measurement noise. It returns new
``PredictionResult`` summaries and leaves the latent prediction summaries intact.

For an AMICI model with normal noise on untransformed observables, obtain the
original observable predictions and their standard deviations using the existing
predictor interface::

    from pypesto.predict import AmiciPredictor
    from pypesto.ensemble import EnsemblePrediction, compute_noisy_summary
    from pypesto.ensemble import get_percentile_label

    predictor = AmiciPredictor(problem.objective)
    predictions = [predictor(x, include_sigmay=True) for x in parameter_points]
    ensemble_prediction = EnsemblePrediction(prediction_results=predictions)
    observation_summary = compute_noisy_summary(
        ensemble_prediction, percentiles=(2.5, 50, 97.5), weights=None
    )
    lower = observation_summary[get_percentile_label(2.5)]
    upper = observation_summary[get_percentile_label(97.5)]

The caller must verify that the noise model is Gaussian and additive in the
same units as the returned outputs. Custom postprocessors, transformed
observables and state predictions require explicit compatible uncertainties;
this function does not infer their noise model. Missing sigma values are errors.

Each condition requires aligned output IDs, condition IDs and timepoints across
members. Outputs and sigmas have exactly the same two-dimensional shape
``(timepoints, outputs)``; no implicit broadcasting occurs. Sigmas can differ
between members, conditions, times and outputs. Only finite outputs and finite
nonnegative sigmas are accepted.

``weights`` are optional nonnegative linear mixture masses. They are not log
likelihoods, and are not automatically taken from ``output_weight``. Equal
weights are used by default. Optimization endpoints do not automatically form
posterior samples; the interpretation of the ensemble must be justified by the
caller.

The observation CDF at each time and output is the weighted sum of the normal
component CDFs. Zero sigma denotes a point mass. Quantiles are computed using
the generalized inverse CDF. With all sigmas zero they reduce to inverse
empirical-CDF quantiles, which deliberately differ from NumPy's default linear
percentile interpolation. Percentile dictionary labels follow the existing
``get_percentile_label`` convention, including the supplied int/float type.
Percentile levels that round to the same dictionary label are rejected.

These intervals are pointwise, not simultaneous trajectory bands. They do not
specify noise correlation across times or outputs. Sensitivities of the new
summaries, lognormal noise, automatic noise-model detection and performance
claims for large ensembles are outside the scope of this prototype.

The existing ``sampling_prediction_trajectories`` plotting function recomputes
the latent summary. It does not consume the dictionary returned here, even if
that dictionary is assigned to ``prediction_summary`` first. Plot the returned
``PredictionResult`` arrays directly. For example, after computing ``lower``
and ``upper`` above, plot one observable in one condition::

    import matplotlib.pyplot as plt

    lower_condition = lower.conditions[0]
    upper_condition = upper.conditions[0]
    plt.fill_between(
        lower_condition.timepoints,
        lower_condition.output[:, 0],
        upper_condition.output[:, 0],
        alpha=0.3,
        label="95% pointwise observation interval",
    )
    plt.legend()

An integrated plotting route is outside this change.

Numerical limitations
---------------------

CDF comparisons use log-domain normal tails, including when checking jumps
from zero-noise atoms. Bisection stops at a representable root or adjacent
floating-point values, rather than after a fixed number of steps. If the
normal tail balance exceeds floating-point precision, the function raises
``ArithmeticError``. A finite output is not silently returned for that case.
Quantiles near cumulative component masses of strongly separated mixtures
remain sensitive to rounding of the supplied weights and probabilities.
