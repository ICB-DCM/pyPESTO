"""Tests for hierarchical optimization with PEtab v2 problems."""

import copy

import numpy as np
import pytest
from amici.sim.sundials import SensitivityMethod
from petab.v1.C import LIN, NOMINAL_VALUE, PARAMETER_SCALE

import pypesto
from pypesto.C import (
    INNER_PARAMETERS,
    MEASUREMENT_TYPE,
    MODE_FUN,
    MODE_RES,
    ORDINAL,
    InnerParameterType,
)
from pypesto.hierarchical import InnerCalculatorCollectorPetabV2
from pypesto.hierarchical.petab import validate_hierarchical_petab_problem
from pypesto.petab import PetabImporter
from pypesto.problem import HierarchicalProblem
from pypesto.testing.examples import (
    get_Boehm_JProteomeRes2014_hierarchical_petab_corrected_bounds,
    get_Boehm_JProteomeRes2014_hierarchical_petab_v2,
)

# tolerances for comparing v1 and v2 objective values and gradients,
# which use different simulator implementations
RTOL_FVAL = 1e-6
RTOL_GRAD = 1e-3


@pytest.fixture(scope="module")
def petab_problem_v1():
    return get_Boehm_JProteomeRes2014_hierarchical_petab_corrected_bounds()


@pytest.fixture(scope="module")
def petab_problem_v2():
    return get_Boehm_JProteomeRes2014_hierarchical_petab_v2()


@pytest.fixture(scope="module")
def importer_v1(petab_problem_v1):
    return PetabImporter(petab_problem_v1, hierarchical=True)


@pytest.fixture(scope="module")
def importer_v2(petab_problem_v2):
    return PetabImporter(
        petab_problem_v2,
        hierarchical=True,
        model_name="Boehm_hierarchical_petab_v2",
    )


@pytest.fixture(scope="module")
def problem_v1(importer_v1):
    return importer_v1.create_problem()


@pytest.fixture(scope="module")
def problem_v2(importer_v2):
    return importer_v2.create_problem()


@pytest.fixture(scope="module")
def objective_v2(importer_v2):
    """A v2 hierarchical objective that is not part of a problem.

    Takes the full vector of outer parameters, including the ones fixed in
    the pypesto problem.
    """
    return importer_v2.create_objective_creator().create_objective()


def _to_linear(x_scaled, scales):
    return np.asarray(
        [
            x if scale == LIN else 10.0**x
            for x, scale in zip(x_scaled, scales, strict=True)
        ]
    )


def _grad_to_scaled(grad_linear, x_linear, scales):
    """Convert a gradient w.r.t. linear parameters to parameter scale."""
    return np.asarray(
        [
            g if scale == LIN else g * x * np.log(10)
            for g, x, scale in zip(grad_linear, x_linear, scales, strict=True)
        ]
    )


def test_hierarchical_petab_v2_structure(
    petab_problem_v2, problem_v1, problem_v2
):
    """The v2 hierarchical problem has the expected structure."""
    assert isinstance(problem_v2, HierarchicalProblem)
    assert isinstance(
        problem_v2.objective.calculator, InnerCalculatorCollectorPetabV2
    )

    # same inner and outer parameters as for PEtab v1
    assert sorted(
        problem_v2.objective.calculator.get_inner_par_ids()
    ) == sorted(problem_v1.objective.calculator.get_inner_par_ids())
    assert problem_v2.x_names == problem_v1.x_names
    assert problem_v2.x_fixed_indices == problem_v1.x_fixed_indices
    assert sorted(problem_v2.inner_x_names) == sorted(problem_v1.inner_x_names)

    # inner parameters are the parameters with a parameterType annotation
    expected_inner_ids = [
        parameter.id
        for parameter in petab_problem_v2.parameters
        if (parameter.model_extra or {}).get("parameterType")
        in list(InnerParameterType)
    ]
    assert sorted(
        problem_v2.objective.calculator.get_inner_par_ids()
    ) == sorted(expected_inner_ids)


def test_hierarchical_petab_v2_matches_v1(
    petab_problem_v1, problem_v1, problem_v2
):
    """Function values, gradients and inner parameters match between the
    PEtab v1 and v2 hierarchical objectives."""
    assert problem_v1.x_free_indices == problem_v2.x_free_indices
    x_names_free = [problem_v1.x_names[ix] for ix in problem_v1.x_free_indices]
    scales = [
        petab_problem_v1.parameter_df.loc[x_id, PARAMETER_SCALE]
        for x_id in x_names_free
    ]
    x_nominal_scaled = np.asarray(
        [
            petab_problem_v1.parameter_df.loc[x_id, NOMINAL_VALUE]
            if scale == LIN
            else np.log10(
                petab_problem_v1.parameter_df.loc[x_id, NOMINAL_VALUE]
            )
            for x_id, scale in zip(x_names_free, scales, strict=True)
        ]
    )

    rng = np.random.default_rng(42)
    for i_point in range(3):
        # the v1 objective operates on scaled parameters, the v2 objective on
        #  linear ones
        x_scaled = x_nominal_scaled + (
            rng.uniform(-0.2, 0.2, size=len(x_names_free)) if i_point else 0.0
        )
        x_linear = _to_linear(x_scaled, scales)

        fval_v1, grad_v1 = problem_v1.objective(x_scaled, sensi_orders=(0, 1))
        fval_v2, grad_v2 = problem_v2.objective(x_linear, sensi_orders=(0, 1))

        assert np.isclose(fval_v1, fval_v2, rtol=RTOL_FVAL)
        assert np.allclose(
            grad_v1,
            _grad_to_scaled(grad_v2, x_linear, scales),
            rtol=RTOL_GRAD,
            atol=RTOL_GRAD * np.max(np.abs(grad_v1)),
        )

        # the optimal inner parameters agree
        ret_v1 = problem_v1.objective(
            x_scaled, sensi_orders=(0,), return_dict=True
        )
        ret_v2 = problem_v2.objective(
            x_linear, sensi_orders=(0,), return_dict=True
        )
        inner_v1 = dict(
            zip(
                problem_v1.inner_x_names,
                ret_v1[INNER_PARAMETERS],
                strict=True,
            )
        )
        inner_v2 = dict(
            zip(
                problem_v2.inner_x_names,
                ret_v2[INNER_PARAMETERS],
                strict=True,
            )
        )
        for inner_id, value_v1 in inner_v1.items():
            assert np.isclose(value_v1, inner_v2[inner_id], rtol=1e-5)


def test_hierarchical_petab_v2_gradient_check(petab_problem_v2, objective_v2):
    """The v2 hierarchical gradient matches finite differences."""
    objective = copy.deepcopy(objective_v2)
    # tight tolerances, so that the finite differences are meaningful also
    #  in flat directions
    objective.amici_solver.set_relative_tolerance(1e-12)
    objective.amici_solver.set_absolute_tolerance(1e-14)

    x_nominal = petab_problem_v2.get_x_nominal_dict()
    x = np.asarray([x_nominal[x_id] for x_id in objective.x_ids])
    # restrict to the parameters estimated in the PEtab problem -- there are
    #  no sensitivities for the others
    x_free_ids = set(petab_problem_v2.x_free_ids)
    x_indices = [
        ix for ix, x_id in enumerate(objective.x_ids) if x_id in x_free_ids
    ]

    check_df = objective.check_grad(
        x, x_indices=x_indices, eps=1e-5, mode=MODE_FUN
    )
    assert np.all(
        (check_df.rel_err.abs() < 1e-3) | (check_df.abs_err < 1e-3)
    ), check_df


def test_hierarchical_petab_v2_adjoint_hessian_residuals(
    petab_problem_v2, objective_v2
):
    """Adjoint sensitivities, the FIM, and residual mode work with the v2
    hierarchical objective and are consistent with the forward results."""
    objective = copy.deepcopy(objective_v2)
    x_nominal = petab_problem_v2.get_x_nominal_dict()
    x = np.asarray([x_nominal[x_id] for x_id in objective.x_ids])

    fval_forward, grad_forward = objective(x, sensi_orders=(0, 1))

    # adjoint sensitivities (computes the inner parameters from a first
    #  simulation, then uses AMICI with fixed inner parameters)
    objective.amici_solver.set_sensitivity_method(SensitivityMethod.adjoint)
    fval_adjoint, grad_adjoint = objective(x, sensi_orders=(0, 1))
    assert np.isclose(fval_forward, fval_adjoint, rtol=1e-6)
    assert np.allclose(
        grad_forward,
        grad_adjoint,
        rtol=1e-3,
        atol=1e-3 * np.max(np.abs(grad_forward)),
    )

    # FIM-based Hessian
    objective.amici_solver.set_sensitivity_method(SensitivityMethod.forward)
    hess = objective(x, sensi_orders=(2,))
    assert hess.shape == (len(x), len(x))
    # the FIM is a positive semi-definite approximation of the Hessian
    eigvals = np.linalg.eigvalsh(hess)
    assert np.min(eigvals) >= -1e-6 * np.max(np.abs(eigvals))

    # residual mode: the residuals of the hierarchical objective are computed
    #  at the optimal inner parameters and are finite
    res = objective(x, sensi_orders=(0,), mode=MODE_RES, return_dict=True)[
        "res"
    ]
    assert np.all(np.isfinite(res))


def test_hierarchical_petab_v2_vs_non_hierarchical(
    petab_problem_v2, objective_v2
):
    """The hierarchical objective matches the non-hierarchical objective
    evaluated at the hierarchically computed optimal inner parameters."""
    importer = PetabImporter(
        petab_problem_v2,
        hierarchical=False,
        model_name="Boehm_hierarchical_petab_v2",
    )
    objective_full = importer.create_objective_creator().create_objective()

    x_nominal = petab_problem_v2.get_x_nominal_dict()
    x_outer = np.asarray([x_nominal[x_id] for x_id in objective_v2.x_ids])

    ret = objective_v2(x_outer, sensi_orders=(0,), return_dict=True)
    fval_hierarchical = ret["fval"]
    inner_parameters = dict(
        zip(
            objective_v2.calculator.get_inner_par_ids(),
            ret[INNER_PARAMETERS],
            strict=True,
        )
    )

    # evaluate the full (non-hierarchical) objective with the inner parameters
    #  set to their hierarchically computed optimal values
    x_full_dict = x_nominal | inner_parameters
    x_full = np.asarray([x_full_dict[x_id] for x_id in objective_full.x_ids])
    fval_full_at_inner_optimum = objective_full(x_full)
    assert np.isclose(fval_hierarchical, fval_full_at_inner_optimum, rtol=1e-6)

    # ... the hierarchical objective is at least as good as the full
    #  objective at the nominal inner parameter values
    x_full_nominal = np.asarray(
        [x_nominal[x_id] for x_id in objective_full.x_ids]
    )
    fval_full_nominal = objective_full(x_full_nominal)
    assert fval_hierarchical <= fval_full_nominal + 1e-8


def test_hierarchical_petab_v2_optimization(petab_problem_v2, problem_v2):
    """Hierarchical optimization of a PEtab v2 problem runs and improves."""
    # start close to the nominal parameters to keep the test fast
    # (some nominal values are on the bounds, so clip the perturbed guesses)
    x_nominal = petab_problem_v2.get_x_nominal_dict()
    x_full = np.asarray([x_nominal[x_id] for x_id in problem_v2.x_names])
    problem_v2.set_x_guesses(
        np.clip(
            np.vstack([x_full * 1.1, x_full * 0.8]),
            problem_v2.lb_full,
            problem_v2.ub_full,
        )
    )

    optimizer = pypesto.optimize.ScipyOptimizer(
        method="L-BFGS-B", options={"maxiter": 10}
    )
    result = pypesto.optimize.minimize(
        problem=problem_v2,
        n_starts=2,
        optimizer=optimizer,
        progress_bar=False,
    )
    for start in result.optimize_result.list:
        assert start.fval0 is not None
        assert start.fval <= start.fval0


def test_hierarchical_petab_v2_validation(petab_problem_v2):
    """Invalid hierarchical PEtab v2 problems are rejected."""
    # the unmodified problem is valid
    validate_hierarchical_petab_problem(petab_problem_v2)

    # unknown parameter type
    petab_problem = copy.deepcopy(petab_problem_v2)
    petab_problem.parameters[-1].model_extra["parameterType"] = "pink"
    with pytest.raises(ValueError, match="Unknown inner parameter type"):
        validate_hierarchical_petab_problem(petab_problem)

    # non-quantitative data types other than relative are not supported yet
    petab_problem = copy.deepcopy(petab_problem_v2)
    petab_problem.measurements[0].model_extra[MEASUREMENT_TYPE] = ORDINAL
    with pytest.raises(NotImplementedError, match="not yet supported"):
        validate_hierarchical_petab_problem(petab_problem)

    # an offset parameter must appear additively in the observable formula
    petab_problem = copy.deepcopy(petab_problem_v2)
    observable = petab_problem.observables[0]
    observable.formula = (
        f"observableParameter1_{observable.id}"
        f" * observableParameter2_{observable.id}"
    )
    with pytest.raises(
        ValueError, match="An offset is in the observable formula"
    ):
        validate_hierarchical_petab_problem(petab_problem)

    # a sigma parameter must constitute the full noise formula
    petab_problem = copy.deepcopy(petab_problem_v2)
    observable = petab_problem.observables[0]
    observable.noise_formula = f"2 * noiseParameter1_{observable.id}"
    with pytest.raises(ValueError, match="full noise formula"):
        validate_hierarchical_petab_problem(petab_problem)

    # non-Gaussian noise distributions are not supported
    petab_problem = copy.deepcopy(petab_problem_v2)
    petab_problem.observables[0].noise_distribution = "laplace"
    with pytest.raises(NotImplementedError, match="oise distribution"):
        validate_hierarchical_petab_problem(petab_problem)


def test_hierarchical_petab_v2_sigma_bounds_are_corrected(petab_problem_v2):
    """Sigma inner parameters get the canonical bounds, whatever the PEtab
    problem declares (PEtab v2 requires bounds, so they cannot be omitted)."""
    from pypesto.C import INNER_PARAMETER_BOUNDS, LOWER_BOUND, UPPER_BOUND
    from pypesto.hierarchical.relative.problem import (
        inner_parameters_from_petab_v2_problem,
    )

    petab_problem = copy.deepcopy(petab_problem_v2)
    for parameter in petab_problem.parameters:
        if parameter.id.startswith("sd_"):
            parameter.lb, parameter.ub = 1e-5, 1e5

    expected = INNER_PARAMETER_BOUNDS[InnerParameterType.SIGMA]
    sigmas = [
        par
        for par in inner_parameters_from_petab_v2_problem(petab_problem)
        if par.inner_parameter_type == InnerParameterType.SIGMA
    ]
    assert sigmas
    for par in sigmas:
        assert par.lb == expected[LOWER_BOUND]
        assert par.ub == expected[UPPER_BOUND]


def test_hierarchical_petab_v2_coupled_pair_rejections(
    petab_problem_v2, importer_v2
):
    """A scaling/offset pair whose partner is not estimated must be rejected
    rather than silently treated as uncoupled, and a numeric override must not
    be counted as a member of the coupled pair."""
    from pypesto.hierarchical.relative.problem import (
        inner_problem_from_petab_v2_problem,
    )

    # the model/edatas of the (already compiled) fixture problem
    creator = importer_v2.create_objective_creator()
    amici_importer = creator._create_amici_importer()
    simulator = amici_importer.create_simulator(force_import=False)
    model = simulator.model
    edatas = simulator.exp_man.create_edatas()
    converted = simulator.exp_man.petab_problem

    # baseline: the unmodified problem couples each scaling with its offset
    inner_problem = inner_problem_from_petab_v2_problem(
        copy.deepcopy(converted), model, edatas
    )
    assert all(
        inner_problem.xs[x_id].coupled is not None
        for x_id in inner_problem.xs
        if inner_problem.xs[x_id].inner_parameter_type
        in (InnerParameterType.SCALING, InnerParameterType.OFFSET)
    )

    # a non-estimated offset partner must raise, not silently uncouple
    petab_problem = copy.deepcopy(converted)
    for parameter in petab_problem.parameters:
        if parameter.id == "offset_pSTAT5A_rel":
            parameter.estimate = False
    with pytest.raises(NotImplementedError, match="not.*estimated"):
        inner_problem_from_petab_v2_problem(petab_problem, model, edatas)

    # ... but if NEITHER is estimated, that observable is simply not
    # hierarchically optimized and the pair must be accepted, not rejected
    petab_problem = copy.deepcopy(converted)
    for parameter in petab_problem.parameters:
        if parameter.id in ("scaling_pSTAT5A_rel", "offset_pSTAT5A_rel"):
            parameter.estimate = False
    inner_problem = inner_problem_from_petab_v2_problem(
        petab_problem, model, edatas
    )
    assert "scaling_pSTAT5A_rel" not in inner_problem.xs
    assert (
        inner_problem.xs["scaling_pSTAT5B_rel"].coupled.inner_parameter_id
        == "offset_pSTAT5B_rel"
    )

    # a numeric override alongside the pair must not inflate the group
    petab_problem = copy.deepcopy(converted)
    for measurement in petab_problem.measurements:
        if measurement.observable_id == "pSTAT5A_rel":
            measurement.observable_parameters = [
                *measurement.observable_parameters,
                1.0,
            ]
    for observable in petab_problem.observables:
        if observable.id == "pSTAT5A_rel":
            observable.observable_placeholders = [
                *map(str, observable.observable_placeholders),
                "observableParameter3_pSTAT5A_rel",
            ]
    inner_problem = inner_problem_from_petab_v2_problem(
        petab_problem, model, edatas
    )
    assert (
        inner_problem.xs["scaling_pSTAT5A_rel"].coupled.inner_parameter_id
        == "offset_pSTAT5A_rel"
    )
