"""Test priors."""

import math

import numpy as np
import pytest

import pypesto
import pypesto.C as C
import pypesto.optimize
from pypesto.C import MODE_FUN, MODE_RES
from pypesto.objective import NegLogParameterPriors
from pypesto.objective.priors import get_parameter_prior_dict
from pypesto.startpoint import UniformStartpoints

scales = [C.LIN, C.LOG, C.LOG10]


@pytest.fixture(params=scales)
def scale(request):
    return request.param


prior_type_lists = [
    [C.UNIFORM],
    [C.NORMAL],
    [C.LAPLACE],
    [C.LOG_NORMAL],
    [C.PARAMETER_SCALE_UNIFORM],
    [C.PARAMETER_SCALE_NORMAL],
    [C.PARAMETER_SCALE_LAPLACE],
    [C.LAPLACE, C.PARAMETER_SCALE_NORMAL, C.PARAMETER_SCALE_LAPLACE],
    [
        C.LAPLACE,
        C.LOG_NORMAL,
        C.PARAMETER_SCALE_NORMAL,
        C.PARAMETER_SCALE_LAPLACE,
    ],
    [
        C.UNIFORM,
        C.NORMAL,
        C.LAPLACE,
        C.LOG_NORMAL,
        C.PARAMETER_SCALE_UNIFORM,
        C.PARAMETER_SCALE_NORMAL,
        C.PARAMETER_SCALE_LAPLACE,
    ],
]


@pytest.fixture(params=prior_type_lists)
def prior_type_list(request):
    return request.param


@pytest.mark.flaky(reruns=3)
def test_mode(scale, prior_type_list):
    """
    Tests the maximum/optimum for priors in different scales...
    """

    problem_dict = {
        "lin": {"lb": 0, "ub": 3, "opt": 1},
        "log": {"lb": -3, "ub": 3, "opt": 0},
        "log10": {"lb": -3, "ub": 2, "opt": 0},
    }

    prior_list = [
        get_parameter_prior_dict(
            iprior,
            prior_type,
            (
                [1, 2]
                if prior_type in [C.UNIFORM, C.PARAMETER_SCALE_UNIFORM]
                else [1, 1]
            ),
            scale,
        )
        for iprior, prior_type in enumerate(prior_type_list)
    ]
    ubs = np.asarray([problem_dict[scale]["ub"] for _ in prior_type_list])
    lbs = np.asarray([problem_dict[scale]["lb"] for _ in prior_type_list])

    test_prior = NegLogParameterPriors(prior_list)
    test_problem = pypesto.Problem(
        test_prior,
        lb=lbs,
        ub=ubs,
        dim_full=len(prior_type_list),
        x_scales=[scale for _ in prior_type_list],
    )

    topt = []
    # test uniform distribution:
    for prior_type, prior in zip(prior_type_list, prior_list, strict=True):
        if prior_type.startswith("parameterScale"):
            scale = C.LIN
        if prior_type in [C.UNIFORM, C.PARAMETER_SCALE_UNIFORM]:
            # check inside and outside of interval
            funprior = prior["density_fun"]
            assert np.isinf(funprior(lin_to_scaled(0.5, scale)))
            assert np.isclose(funprior(lin_to_scaled(1.5, scale)), math.log(1))
            assert np.isinf(funprior(lin_to_scaled(2.5, scale)))
            resprior = prior["residual"]
            assert np.isinf(resprior(lin_to_scaled(0.5, scale)))
            assert np.isclose(resprior(lin_to_scaled(1.5, scale)), 0)
            assert np.isinf(resprior(lin_to_scaled(2.5, scale)))
            topt.append(np.nan)
        else:
            topt.append(problem_dict[scale]["opt"])

        if prior_type.endswith("logNormal"):
            assert not test_prior.has_res
            assert not test_prior.has_sres

    topt = np.asarray(topt)

    # test log-density based and residual representation
    if any(~np.isnan(topt)):
        for method in ["L-BFGS-B", "ls_trf"]:
            if method == "ls_trf" and not test_prior.has_res:
                continue
            optimizer = pypesto.optimize.ScipyOptimizer(method=method)
            test_problem.startpoint_method = UniformStartpoints(
                check_fval=True,
            )
            result = pypesto.optimize.minimize(
                problem=test_problem,
                optimizer=optimizer,
                n_starts=10,
                progress_bar=False,
            )

            # flat functions don't have local minima, so don't check this
            # for uniform priors

            num_optim = result.optimize_result.list[0]["x"][~np.isnan(topt)]
            assert np.isclose(
                num_optim, topt[~np.isnan(topt)], atol=1e-03
            ).all()


def test_derivatives(prior_type_list, scale):
    """
    Tests the finite gradients and second order derivatives.
    """

    prior_list = [
        get_parameter_prior_dict(
            iprior,
            prior_type,
            (
                [-1, 1]
                if prior_type in [C.UNIFORM, C.PARAMETER_SCALE_UNIFORM]
                else [1, 1]
            ),
            scale,
        )
        for iprior, prior_type in enumerate(prior_type_list)
    ]

    test_prior = NegLogParameterPriors(prior_list)

    # use this x0, since it is a moderate value both in linear
    # and in log scale...
    x0 = np.array([lin_to_scaled(0.5, scale)] * len(prior_list))

    multi_eps = [1e-3]
    assert test_prior.check_gradients_match_finite_differences(
        x=x0, mode=MODE_FUN, multi_eps=multi_eps
    )
    assert test_prior.check_gradients_match_finite_differences(
        x=x0, mode=MODE_FUN, order=1, multi_eps=multi_eps
    )

    if test_prior.has_res:
        test_prior.check_gradients_match_finite_differences(
            x=x0, mode=MODE_RES, multi_eps=multi_eps
        )


@pytest.mark.flaky(reruns=3)
def test_sample(prior_type_list, scale):
    """
    Tests sampling from the priors and compare to the density.
    """
    np.random.seed(0)

    prior_list = [
        get_parameter_prior_dict(
            iprior,
            prior_type,
            (
                [9, 10]
                if prior_type in [C.UNIFORM, C.PARAMETER_SCALE_UNIFORM]
                else (
                    [0.5, 0.1]
                    if prior_type in [C.LOG_NORMAL, C.PARAMETER_SCALE_NORMAL]
                    else [10, 1]
                )
            ),
            scale,
        )
        for iprior, prior_type in enumerate(prior_type_list)
    ]

    test_prior = NegLogParameterPriors(prior_list)

    n_samples = 1000
    sample_dict = test_prior.sample(n_samples=n_samples)

    for iprior, prior in enumerate(prior_list):
        samples = np.asarray(sample_dict[iprior])
        # basic sanity check
        assert samples.shape == (n_samples,)
        # empirical pdf from samples
        xs, pdf_emp = _empirical_pdf_from_samples(samples)
        pdf_emp_m = pdf_emp / np.sum(pdf_emp)

        # theoretical log-density function in the same space as the samples
        logpdf = prior["density_fun"]
        logpdf_vals = np.array([logpdf(x) for x in xs])
        pdf_theo = np.exp(logpdf_vals)

        # Relative L2 error of empirical vs theoretical density
        num = np.linalg.norm(pdf_emp_m - pdf_theo)
        den = np.linalg.norm(pdf_theo)
        rel_err = num / den if den > 0 else num
        assert rel_err < 1, (
            f"Empirical vs theoretical pdf mismatch for prior {iprior} "
            f"(relative L2 error = {rel_err:.3f})"
        )


def lin_to_scaled(x: float, scale: str):
    """
    transforms x to linear scale
    """
    if scale == "lin":
        return x
    elif scale == "log":
        return math.log(x)
    elif scale == "log10":
        return math.log10(x)
    else:
        ValueError(f"Unknown scale {scale}")


def scaled_to_lin(x: float, scale: str):
    """
    transforms x to scale
    """
    if scale == "lin":
        return x
    elif scale == "log":
        return math.exp(x)
    elif scale == "log10":
        return 10**x
    else:
        ValueError(f"Unknown scale {scale}")


def _empirical_pdf_from_samples(samples, n_bins=100):
    """
    Compute an empirical PDF estimate from samples using a histogram.

    Returns
    -------
    bin_centers : np.ndarray
        Centers of the histogram bins.
    pdf_emp : np.ndarray
        Estimated probability density at the bin centers.
    """
    samples = np.asarray(samples)
    counts, bin_edges = np.histogram(samples, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, counts
