"""Tests for the observable formula forms of hierarchical optimization.

An observable ``y = scaling * h + constant [+ offset]`` is simulated with the
scaling at its dummy value 1, i.e. as ``h + constant``. The inner problem then
has to subtract the constant again, see #1785.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import petab.v1 as petab
import pytest
import sympy as sp
from amici.importers.antimony import antimony2sbml
from amici.sim.sundials import SensitivityMethod
from petab import v2

from pypesto.C import (
    AMICI_Y,
    FVAL,
    GRAD,
    INNER_PARAMETER_BOUNDS,
    INNER_PARAMETERS,
    LIN,
    LOWER_BOUND,
    PARAMETER_TYPE,
    RDATAS,
    UPPER_BOUND,
    InnerParameterType,
)
from pypesto.hierarchical.base_parameter import InnerParameter
from pypesto.hierarchical.petab import (
    _validate_observable_formula_form,
    get_unscaled_constants,
    validate_hierarchical_petab_problem,
)
from pypesto.hierarchical.relative import (
    AnalyticalInnerSolver,
    NumericalInnerSolver,
    RelativeInnerProblem,
)
from pypesto.petab import PetabImporter

SCALING = InnerParameterType.SCALING
OFFSET = InnerParameterType.OFFSET

s, b, t = sp.symbols("s b t")


@pytest.mark.parametrize(
    "formula, inner_parameters, expected",
    [
        # (formula, inner parameters, (offset, scaling, constant))
        ("s*x1", {s: SCALING}, (None, s, 0)),
        ("s*(x1 + x2)", {s: SCALING}, (None, s, 0)),
        ("s*x1 + b", {s: SCALING, b: OFFSET}, (b, s, 0)),
        ("s*x1 + 1", {s: SCALING}, (None, s, 1)),
        ("s*(x1 + x2) + 1", {s: SCALING}, (None, s, 1)),
        ("2*s*x1 + 3", {s: SCALING}, (None, s, 3)),
        ("3 - s*x1", {s: SCALING}, (None, s, 3)),
        ("s*x1/(x2 + 1) + 0.25", {s: SCALING}, (None, s, 0.25)),
        ("s*(x2 + 2) - 1", {s: SCALING}, (None, s, -1)),
        ("s*x1 + 1 + b", {s: SCALING, b: OFFSET}, (b, s, 1)),
        ("s*x1 + 0.5 + 2", {s: SCALING}, (None, s, 2.5)),
        ("s*x1 + pi", {s: SCALING}, (None, s, np.pi)),
        # an offset alone may come with any other terms
        ("x1 + x2 + b", {b: OFFSET}, (b, None, 0)),
        ("x1 + 1 + b", {b: OFFSET}, (b, None, 0)),
        # terms that the scaling does not multiply must be constant
        ("s*(x1 + x2) + x1", {s: SCALING}, ValueError),
        ("s*x1 + x2 + b", {s: SCALING, b: OFFSET}, ValueError),
        ("s*x1 + p", {s: SCALING}, ValueError),
        ("s*x1 + t", {s: SCALING}, ValueError),
        # the scaling must be a single factor of a single term
        ("s*(x1 + x2) + s", {s: SCALING}, ValueError),
        ("s*x1 + s*x2", {s: SCALING}, ValueError),
        ("s**2*(x1 + x2)", {s: SCALING}, ValueError),
        ("x1/s", {s: SCALING}, ValueError),
        ("exp(s)*x1", {s: SCALING}, ValueError),
        # the offset must be its own term
        ("s*(x1 + b)", {s: SCALING, b: OFFSET}, ValueError),
        ("s*x1 + 2*b", {s: SCALING, b: OFFSET}, ValueError),
    ],
)
def test_observable_formula_forms(formula, inner_parameters, expected):
    """Supported forms are split into offset, scaling and constant; others
    are rejected."""
    formula = sp.sympify(formula, locals={"t": t})
    if expected is ValueError:
        with pytest.raises(ValueError):
            _validate_observable_formula_form(formula, inner_parameters)
        return
    offset, scaling, constant = _validate_observable_formula_form(
        formula, inner_parameters
    )
    assert (offset, scaling) == expected[:2]
    assert np.isclose(constant, expected[2], rtol=0, atol=1e-14)


def _inner_parameter(inner_parameter_id, inner_parameter_type, mask):
    return InnerParameter(
        inner_parameter_id=inner_parameter_id,
        inner_parameter_type=inner_parameter_type,
        scale=LIN,
        lb=INNER_PARAMETER_BOUNDS[inner_parameter_type][LOWER_BOUND],
        ub=INNER_PARAMETER_BOUNDS[inner_parameter_type][UPPER_BOUND],
        ixs=mask,
    )


@pytest.mark.parametrize("with_offset", [False, True])
@pytest.mark.parametrize(
    "solver_class", [AnalyticalInnerSolver, NumericalInnerSolver]
)
def test_inner_solver_with_unscaled_constants(solver_class, with_offset):
    """With unscaled constants, the inner solvers give the inner parameters
    and objective of the problem without them.

    Two observables share a scaling, with different constants, and only the
    first one is measured at every time point.
    """
    rng = np.random.default_rng(0)
    h = np.exp(-np.linspace(0, 3, 21))[:, None] * np.array([[1.0, 2.0]])
    constants = np.array([[1.5, -0.5]]) * np.ones_like(h)
    scaling, offset = 3.0, 0.7 if with_offset else 0.0
    data = scaling * h + constants + offset + rng.normal(0, 0.1, h.shape)
    data[::2, 1] = np.nan
    mask = [~np.isnan(data)]

    def inner_problem(unscaled_constants):
        xs = [_inner_parameter("scaling_", SCALING, mask)]
        if with_offset:
            xs.append(_inner_parameter("offset_", OFFSET, mask))
            xs[0].coupled, xs[1].coupled = xs[1], xs[0]
        return RelativeInnerProblem(
            xs=xs,
            data=[data - (constants if unscaled_constants is None else 0)],
            edatas=None,
            unscaled_constants=unscaled_constants,
        )

    # the same problem, once simulated with the constants, once without
    with_constants = inner_problem([constants])
    reference = inner_problem(None)
    sim, sigma = [h + constants], [np.ones_like(h)]

    solver = solver_class()
    x = solver.solve(with_constants, sim=sim, sigma=sigma, scaled=False)
    x_ref = solver_class().solve(reference, sim=[h], sigma=sigma, scaled=False)
    for x_id, value in x_ref.items():
        assert np.isclose(x[x_id], value, rtol=1e-6), x_id
    assert np.isclose(x["scaling_"], scaling, rtol=0.05)

    assert np.isclose(
        solver.calculate_obj_function(with_constants, sim, sigma, x),
        solver.calculate_obj_function(reference, [h], sigma, x_ref),
        rtol=1e-10,
    )

    # applying the inner parameters gives the model output of the measured
    #  observables
    rdatas = [{"y": copy.deepcopy(sim[0]), "sigmay": copy.deepcopy(sigma[0])}]
    solver.apply_inner_parameters_to_rdatas(with_constants, rdatas, x)
    assert np.allclose(
        rdatas[0]["y"][mask[0]],
        (x["scaling_"] * h + constants + x.get("offset_", 0))[mask[0]],
        rtol=1e-12,
    )


def test_get_unscaled_constants_of_fixed_scaling():
    """A scaling that is annotated, but not estimated, is no inner parameter:
    the model is simulated at its nominal value, so the other terms of the
    observable formula are not restricted."""
    petab_problem = petab.Problem(
        observable_df=petab.get_observable_df(
            pd.DataFrame(
                {
                    petab.OBSERVABLE_ID: ["obs"],
                    petab.OBSERVABLE_FORMULA: [
                        "observableParameter1_obs * x1 + x2"
                    ],
                    petab.NOISE_FORMULA: [0.1],
                }
            )
        ),
        measurement_df=petab.get_measurement_df(
            pd.DataFrame(
                {
                    petab.OBSERVABLE_ID: ["obs"],
                    petab.SIMULATION_CONDITION_ID: ["c0"],
                    petab.TIME: [0],
                    petab.MEASUREMENT: [1],
                    petab.OBSERVABLE_PARAMETERS: ["s"],
                }
            )
        ),
        parameter_df=petab.get_parameter_df(
            pd.DataFrame(
                {
                    petab.PARAMETER_ID: ["s"],
                    PARAMETER_TYPE: [SCALING],
                    petab.ESTIMATE: [0],
                }
            )
        ),
    )
    assert get_unscaled_constants(petab_problem, inner_parameter_ids=[]) == [
        0.0
    ]
    with pytest.raises(ValueError, match="not constant"):
        get_unscaled_constants(petab_problem, inner_parameter_ids=["s"])


#: The observables of the toy model, one per form. Each maps to its
#: observable formula, its noise formula, and its observable and noise
#: parameter overrides in each condition.
OBSERVABLES = {
    # proportional, as a control
    "obs_prop": (
        "observableParameter1_obs_prop * x1",
        "0.1",
        ["s_prop"] * 2,
        None,
    ),
    # the form of #1785
    "obs_const": (
        "observableParameter1_obs_const * x1 + 1",
        "0.1",
        ["s_const"] * 2,
        None,
    ),
    # a constant and an offset
    "obs_const_offset": (
        "observableParameter1_obs_const_offset * (x1 + x2) + 0.5 + observableParameter2_obs_const_offset",
        "0.1",
        ["s_co;b_co"] * 2,
        None,
    ),
    # the constant is a numeric override; two observables with different
    #  constants share the scaling. (PEtab v2 import in AMICI does not
    #  support overrides that differ between conditions.)
    "obs_override": (
        "observableParameter1_obs_override * x2"
        " + observableParameter2_obs_override",
        "0.1",
        ["s_ov;0.3"] * 2,
        None,
    ),
    "obs_override2": (
        "observableParameter1_obs_override2 * x1"
        " + observableParameter2_obs_override2",
        "0.1",
        ["s_ov;1.2"] * 2,
        None,
    ),
    # the scaled term contains a constant too
    "obs_scaled_const": (
        "observableParameter1_obs_scaled_const * (x2 + 2) - 1",
        "0.1",
        ["s_sc"] * 2,
        None,
    ),
    # a rational function, with a constant in front, and a negative sign
    "obs_rational": (
        "4 - observableParameter1_obs_rational * x1 / (x2 + 1)",
        "0.1",
        ["s_rat"] * 2,
        None,
    ),
    # a constant and a hierarchical sigma
    "obs_sigma": (
        "observableParameter1_obs_sigma * x1 * x2 + 3",
        "noiseParameter1_obs_sigma",
        ["s_sig"] * 2,
        ["sd_sig"] * 2,
    ),
    # an offset alone, with other (non-constant) terms
    "obs_offset_only": (
        "x1 + x2 + 2 + observableParameter1_obs_offset_only",
        "0.1",
        ["b_oo"] * 2,
        None,
    ),
}

#: inner parameters, their types and the values the data are generated with
INNER_PARAMETERS_TRUE = {
    "s_prop": (SCALING, 2.0),
    "s_const": (SCALING, 2.0),
    "s_co": (SCALING, 1.5),
    "b_co": (OFFSET, 0.4),
    "s_ov": (SCALING, 3.0),
    "s_sc": (SCALING, 0.8),
    "s_rat": (SCALING, 1.7),
    "s_sig": (SCALING, 5.0),
    "sd_sig": (InnerParameterType.SIGMA, 0.2),
    "b_oo": (OFFSET, -0.3),
}
OUTER_PARAMETERS_TRUE = {"k1": 0.5, "k2": 0.3}
TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
CONDITIONS = ["c0", "c1"]


#: x1' = -k1 x1, x2' = k1 x1 - k2 x2, with x1(0) = 1, x2(0) = 0
TOY_MODEL = """
model toy
  compartment cell = 1
  species x1 in cell = 1, x2 in cell = 0
  J1: x1 -> x2; k1 * x1
  J2: x2 -> ; k2 * x2
  k1 = 0.5
  k2 = 0.3
end
"""


def _true_measurements(observable_id: str, condition_ix: int) -> np.ndarray:
    """The observable at the true parameters, at ``TIMES``."""
    k1, k2 = OUTER_PARAMETERS_TRUE.values()
    states = {
        "x1": np.exp(-k1 * TIMES),
        "x2": k1 / (k2 - k1) * (np.exp(-k1 * TIMES) - np.exp(-k2 * TIMES)),
    }
    formula, _, overrides, _ = OBSERVABLES[observable_id]
    formula = sp.sympify(formula)
    placeholders = sorted(
        (
            symbol
            for symbol in formula.free_symbols
            if symbol.name.startswith("observableParameter")
        ),
        key=lambda symbol: symbol.name,
    )
    values = {
        par_id: value for par_id, (_, value) in INNER_PARAMETERS_TRUE.items()
    }
    formula = formula.subs(
        {
            placeholder: values.get(override, override)
            for placeholder, override in zip(
                placeholders,
                overrides[condition_ix].split(petab.PARAMETER_SEPARATOR),
                strict=True,
            )
        }
    )
    return sp.lambdify(list(states), formula)(*states.values()) * np.ones(
        TIMES.size
    )


@pytest.fixture(scope="module")
def toy_problem_dir(tmp_path_factory) -> Path:
    """A PEtab v1 problem with one observable per form, with noisy data."""
    directory = tmp_path_factory.mktemp("observable_formula_forms")
    (directory / "model.xml").write_text(antimony2sbml(TOY_MODEL))

    rng = np.random.default_rng(1)
    measurements = []
    for observable_id, (
        _,
        _,
        overrides,
        noise_overrides,
    ) in OBSERVABLES.items():
        for condition_ix, condition_id in enumerate(CONDITIONS):
            sigma = 0.2 if noise_overrides else 0.1
            measurements.append(
                pd.DataFrame(
                    {
                        petab.OBSERVABLE_ID: observable_id,
                        petab.SIMULATION_CONDITION_ID: condition_id,
                        petab.TIME: TIMES,
                        petab.MEASUREMENT: _true_measurements(
                            observable_id, condition_ix
                        )
                        + rng.normal(0, sigma, TIMES.size),
                        petab.OBSERVABLE_PARAMETERS: overrides[condition_ix],
                        petab.NOISE_PARAMETERS: (
                            noise_overrides[condition_ix]
                            if noise_overrides
                            else np.nan
                        ),
                    }
                )
            )

    inner_ids = list(INNER_PARAMETERS_TRUE)
    tables = {
        "conditions": pd.DataFrame(
            {petab.CONDITION_ID: CONDITIONS, petab.CONDITION_NAME: CONDITIONS}
        ),
        "observables": pd.DataFrame(
            {
                petab.OBSERVABLE_ID: list(OBSERVABLES),
                petab.OBSERVABLE_FORMULA: [
                    formula for formula, *_ in OBSERVABLES.values()
                ],
                petab.NOISE_FORMULA: [
                    noise_formula
                    for _, noise_formula, *_ in OBSERVABLES.values()
                ],
            }
        ),
        "measurements": pd.concat(measurements, ignore_index=True),
        "parameters": pd.DataFrame(
            {
                petab.PARAMETER_ID: list(OUTER_PARAMETERS_TRUE) + inner_ids,
                petab.PARAMETER_SCALE: LIN,
                # hierarchical sigmas need the bounds [0, inf]
                petab.LOWER_BOUND: [0.01, 0.01]
                + [
                    {SCALING: 0.01, OFFSET: -100}.get(parameter_type, 0)
                    for parameter_type, _ in INNER_PARAMETERS_TRUE.values()
                ],
                petab.UPPER_BOUND: [100.0, 100.0]
                + [
                    np.inf
                    if parameter_type == InnerParameterType.SIGMA
                    else 100.0
                    for parameter_type, _ in INNER_PARAMETERS_TRUE.values()
                ],
                petab.NOMINAL_VALUE: list(OUTER_PARAMETERS_TRUE.values())
                + [value for _, value in INNER_PARAMETERS_TRUE.values()],
                petab.ESTIMATE: 1,
                PARAMETER_TYPE: [None, None]
                + [
                    parameter_type
                    for parameter_type, _ in INNER_PARAMETERS_TRUE.values()
                ],
            }
        ),
    }
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.tsv", sep="\t", index=False)
    (directory / "problem.yaml").write_text(
        "format_version: 1\n"
        "parameter_file: parameters.tsv\n"
        "problems:\n"
        "- condition_files: [conditions.tsv]\n"
        "  measurement_files: [measurements.tsv]\n"
        "  observable_files: [observables.tsv]\n"
        "  sbml_files: [model.xml]\n"
    )
    return directory


def _load(toy_problem_dir: Path, petab_version: int):
    yaml_file = toy_problem_dir / "problem.yaml"
    if petab_version == 1:
        return petab.Problem.from_yaml(yaml_file)
    # upgrades the problem to PEtab v2, keeping the `parameterType` column
    return v2.Problem.from_yaml(yaml_file)


@pytest.fixture(scope="module", params=[1, 2])
def objectives(request, toy_problem_dir, tmp_path_factory):
    """The hierarchical and the standard objective for a PEtab version."""
    petab_problem = _load(toy_problem_dir, request.param)
    model_name = f"observable_formula_forms_v{request.param}"
    output_folder = tmp_path_factory.mktemp("amici_models") / model_name
    objectives = {}
    for hierarchical in (True, False):
        importer = PetabImporter(
            copy.deepcopy(petab_problem),
            hierarchical=hierarchical,
            model_name=model_name,
            output_folder=str(output_folder),
        )
        objectives[hierarchical] = (
            importer.create_objective_creator().create_objective(verbose=False)
        )
    return objectives


@pytest.mark.parametrize(
    "outer_parameters",
    [OUTER_PARAMETERS_TRUE, {"k1": 0.7, "k2": 0.2}],
    ids=["true", "perturbed"],
)
@pytest.mark.parametrize(
    "sensitivity_method",
    [SensitivityMethod.forward, SensitivityMethod.adjoint],
    ids=["forward", "adjoint"],
)
def test_observable_formula_forms_objective(
    objectives, outer_parameters, sensitivity_method
):
    """The hierarchical objective is the standard objective minimized over
    the inner parameters, for every supported form.

    At the inner parameters computed by the hierarchical objective, the
    standard objective has the same value, its gradient with respect to the
    inner parameters vanishes, and its gradient with respect to the outer
    parameters is the hierarchical gradient. Also, the simulations of both
    agree.
    """
    hierarchical, standard = objectives[True], objectives[False]
    for objective in (hierarchical, standard):
        objective.amici_solver.set_sensitivity_method(sensitivity_method)

    x_outer = np.asarray(
        [outer_parameters[x_id] for x_id in hierarchical.x_ids]
    )
    ret_hierarchical = hierarchical(
        x_outer, sensi_orders=(0, 1), return_dict=True
    )
    inner_ids = hierarchical.calculator.get_inner_par_ids()
    assert sorted(inner_ids) == sorted(INNER_PARAMETERS_TRUE)
    inner_parameters = dict(
        zip(inner_ids, ret_hierarchical[INNER_PARAMETERS], strict=True)
    )

    x_full = outer_parameters | inner_parameters
    ret_standard = standard(
        np.asarray([x_full[x_id] for x_id in standard.x_ids]),
        sensi_orders=(0, 1),
        return_dict=True,
    )
    gradient = dict(zip(standard.x_ids, ret_standard[GRAD], strict=True))

    assert np.isclose(ret_hierarchical[FVAL], ret_standard[FVAL], rtol=1e-8), (
        ret_hierarchical[FVAL] - ret_standard[FVAL]
    )
    for inner_id in inner_ids:
        assert np.isclose(gradient[inner_id], 0, atol=1e-4), (
            inner_id,
            gradient[inner_id],
        )
    assert np.allclose(
        ret_hierarchical[GRAD],
        [gradient[x_id] for x_id in hierarchical.x_ids],
        rtol=1e-4,
        atol=1e-6,
    )

    # the simulations are at the optimal inner parameters
    for rdata_hierarchical, rdata_standard in zip(
        ret_hierarchical[RDATAS], ret_standard[RDATAS], strict=True
    ):
        assert np.allclose(
            rdata_hierarchical[AMICI_Y],
            rdata_standard[AMICI_Y],
            rtol=1e-8,
            atol=1e-10,
        )

    if outer_parameters == OUTER_PARAMETERS_TRUE:
        # close to the values that the data were generated with
        for inner_id, (_, value) in INNER_PARAMETERS_TRUE.items():
            assert np.isclose(
                inner_parameters[inner_id], value, rtol=0.2, atol=0.2
            ), (inner_id, inner_parameters[inner_id], value)


@pytest.mark.parametrize("petab_version", [1, 2])
def test_validate_non_constant_unscaled_terms(toy_problem_dir, petab_version):
    """Terms that the scaling does not multiply, and that are not constant,
    are rejected."""
    petab_problem = _load(toy_problem_dir, petab_version)
    validate_hierarchical_petab_problem(petab_problem)

    if petab_version == 1:
        petab_problem.observable_df.loc[
            "obs_const", petab.OBSERVABLE_FORMULA
        ] += " + x2"
    else:
        petab_problem["obs_const"].formula += sp.Symbol("x2", real=True)
    with pytest.raises(ValueError, match="Non-constant terms: `x2 ?[+]"):
        validate_hierarchical_petab_problem(petab_problem)
