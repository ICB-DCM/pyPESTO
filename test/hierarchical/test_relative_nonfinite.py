"""Measured model failures must not become missing relative measurements."""

import math

import numpy as np
import pytest

from pypesto.hierarchical.relative.util import (
    compute_nllh,
    compute_optimal_sigma,
)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("position", range(3))
def test_nonfinite_model_output_is_not_missing(bad, position):
    data = [np.array([[1.1], [1.8], [3.2]])]
    sim = [np.array([[1.0], [2.0], [3.0]])]
    sim[0][position, 0] = bad
    mask = [np.ones((3, 1), dtype=bool)]
    assert not np.isfinite(compute_optimal_sigma(data, sim, mask))
    assert not np.isfinite(compute_nllh(data, sim, [np.ones((3, 1))], mask))


def test_unmeasured_nonfinite_values_are_ignored():
    data = [np.array([[np.nan, 2.2], [0.4, np.nan]])]
    sim = [np.array([[np.nan, 2.0], [0.0, np.inf]])]
    mask = [np.array([[False, True], [True, False]])]
    variance = (0.2**2 + 0.4**2) / 2
    sigma = compute_optimal_sigma(data, sim, mask)
    assert sigma == pytest.approx(math.sqrt(variance))
    actual = compute_nllh(data, sim, [np.full((2, 2), sigma)], mask)
    assert actual == pytest.approx(1 + math.log(2 * math.pi * variance))


def test_empty_measurement_mask_has_zero_likelihood_sum():
    shape = (2, 2)
    assert (
        compute_nllh(
            [np.full(shape, np.nan)],
            [np.full(shape, np.nan)],
            [np.full(shape, np.nan)],
            [np.zeros(shape, dtype=bool)],
        )
        == 0
    )


@pytest.fixture(
    scope="module", params=["invalid", "unmeasured", "finite", "zero"]
)
def public_problem(request, tmp_path_factory):
    """Use actual PEtab import, compiled AMICI model and public collector."""
    pytest.importorskip("amici.sim.sundials")
    pytest.importorskip("antimony")
    petab = pytest.importorskip("petab.v2")
    from petab.v2.models.sbml_model import SbmlModel

    from pypesto.C import PARAMETER_TYPE, InnerParameterType
    from pypesto.petab import PetabImporter

    kind = request.param
    x0, y0 = (
        (1.0, 1.0)
        if kind == "finite"
        else (0.0, 1.0)
        if kind == "zero"
        else (0.0, 0.0)
    )
    p = petab.Problem()
    p.model = SbmlModel.from_antimony(
        f"rate=1; x={x0}; y={y0}; x'=rate; y'=1;"
    )
    p.add_parameter("rate", nominal_value=1, estimate=True, lb=0, ub=2)
    p.add_parameter(
        "sigma",
        nominal_value=1,
        estimate=True,
        lb=0,
        ub=float("inf"),
        **{PARAMETER_TYPE: InnerParameterType.SIGMA},
    )
    p.add_observable(
        "ratio",
        formula="x/(x+y)",
        noise_formula="noiseParameter1_ratio",
        noise_placeholders=["noiseParameter1_ratio"],
    )
    p.add_observable("plain", formula="x", noise_formula="1")
    p.add_experiment("exp", 0, [])
    measurements = [(0.0, 0.6), (1.0, 0.4), (2.0, 0.7)]
    if kind == "unmeasured":
        measurements = measurements[1:]
    for time, value in measurements:
        p.add_measurement(
            "ratio",
            experiment_id="exp",
            time=time,
            measurement=value,
            noise_parameters="sigma",
        )
    p.add_measurement(
        "plain", experiment_id="exp", time=0, measurement=x0 + 0.1
    )
    output = tmp_path_factory.mktemp(f"relative_nonfinite_{kind}") / "model"
    problem = PetabImporter(
        p,
        hierarchical=True,
        model_name=f"relative_nonfinite_{kind}",
        output_folder=str(output),
    ).create_problem()
    return kind, problem, x0, y0, measurements


@pytest.mark.parametrize(
    "orders", [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
)
def test_public_hierarchical_nan_contract(public_problem, orders):
    kind, problem, x0, y0, measurements = public_problem
    rate = 0.0 if kind == "zero" else 1.0
    result = problem.objective(
        np.array([rate]), sensi_orders=orders, return_dict=True
    )
    assert all(rdata.status == 0 for rdata in result["rdatas"])
    if kind == "invalid":
        assert np.isinf(result["fval"])
        if 1 in orders:
            assert np.isnan(result["grad"]).all()
        if 2 in orders:
            assert np.isnan(result["hess"]).all()
        return

    # Independent ordinary Gaussian likelihood and envelope gradient for the
    # analytically known solution x=x0+rate*t, y=y0+t, plus a plain measurement.
    times, data = map(np.asarray, zip(*measurements, strict=True))
    denominator = x0 + y0 + (rate + 1) * times
    prediction = (x0 + rate * times) / denominator
    derivative = times * (y0 + times) / denominator**2
    variance = np.mean((prediction - data) ** 2)
    expected = len(data) / 2 * (1 + np.log(2 * np.pi * variance))
    expected += 0.5 * np.log(2 * np.pi) + 0.5 * 0.1**2
    if 0 in orders:
        assert result["fval"] == pytest.approx(expected, rel=1e-7, abs=1e-9)
    if 1 in orders:
        gradient = np.sum((prediction - data) * derivative) / variance
        assert result["grad"][0] == pytest.approx(gradient, rel=1e-6, abs=1e-8)
    if 2 in orders:
        # The configured second-order return is the conditional FIM.
        assert result["hess"][0, 0] == pytest.approx(
            np.sum(derivative**2) / variance, rel=1e-6, abs=1e-8
        )
