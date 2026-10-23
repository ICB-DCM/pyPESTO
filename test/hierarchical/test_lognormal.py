"""Native PEtab-v1 log-normal inner solves against standard AMICI."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import petab.v1 as petab
import pytest
from amici.sim.sundials import SensitivityMethod
from petab.v1.models.sbml_model import SbmlModel

from pypesto.C import (
    FVAL,
    GRAD,
    INNER_PARAMETERS,
    MODE_FUN,
    MODE_RES,
    InnerParameterType,
)
from pypesto.hierarchical.petab import validate_hierarchical_petab_problem
from pypesto.optimize import ScipyOptimizer, minimize
from pypesto.petab import PetabImporter


def make_problem(base, inner="both", bounded=False):
    """Two conditions, condition-specific nuisances, replicates and control."""
    model = SbmlModel.from_antimony(
        """
        model lognormal_conversion
            A = 1; B = 0.1; k1 = 0.5; k2 = 0.1;
            r1: A => B; k1*A;
            r2: B => A; k2*B;
        end
        """
    )
    observable = pd.DataFrame(
        {
            petab.OBSERVABLE_ID: ["relative", "quantitative"],
            petab.OBSERVABLE_FORMULA: [
                "observableParameter1_relative*A",
                "B+0.5",
            ],
            petab.NOISE_FORMULA: [
                "noiseParameter1_relative",
                "noiseParameter1_quantitative",
            ],
            petab.OBSERVABLE_TRANSFORMATION: [base, petab.LIN],
            petab.NOISE_DISTRIBUTION: petab.NORMAL,
        }
    )
    rows = []
    times = np.array([0.0, 0.5, 1.0, 1.0, 2.0, 4.0, 7.0])
    for condition, initial in enumerate([1.0, 2.0], start=1):
        equilibrium = (initial + 0.1) / 6
        a = equilibrium + (initial - equilibrium) * np.exp(-0.6 * times)
        for ix, (time, prediction) in enumerate(zip(times, a, strict=True)):
            rows.append(
                {
                    petab.OBSERVABLE_ID: "relative",
                    petab.SIMULATION_CONDITION_ID: f"c{condition}",
                    petab.TIME: time,
                    petab.MEASUREMENT: (1.3 + condition * 0.2)
                    * prediction
                    * np.exp(0.13 * np.sin(ix + condition)),
                    petab.OBSERVABLE_PARAMETERS: f"s{condition}",
                    petab.NOISE_PARAMETERS: f"q{condition}",
                }
            )
            # Quantitative measurements only at alternate times. The other
            # observation slots are missing in the AMICI measurement matrix.
            if ix % 2 == 0:
                rows.append(
                    {
                        petab.OBSERVABLE_ID: "quantitative",
                        petab.SIMULATION_CONDITION_ID: f"c{condition}",
                        petab.TIME: time,
                        petab.MEASUREMENT: initial + 0.6 - prediction + 0.03,
                        petab.OBSERVABLE_PARAMETERS: "",
                        petab.NOISE_PARAMETERS: "0.2",
                    }
                )
    parameter = pd.DataFrame(
        {
            petab.PARAMETER_ID: ["k1", "k2", "s1", "s2", "q1", "q2"],
            petab.PARAMETER_SCALE: [
                petab.LOG10,
                petab.LIN,
                petab.LIN,
                petab.LOG10,
                petab.LIN,
                petab.LIN,
            ],
            petab.LOWER_BOUND: [0.001, 0.001, 0.2, 0.2, 0.0, 0.0],
            petab.UPPER_BOUND: [
                10.0,
                10.0,
                0.4 if bounded else 5.0,
                5.0,
                np.inf,
                np.inf,
            ],
            petab.NOMINAL_VALUE: [
                0.4,
                0.15,
                0.3 if bounded else 1.0,
                1.0,
                0.15,
                0.15,
            ],
            petab.ESTIMATE: [
                1,
                1,
                int(inner != "sigma"),
                int(inner != "sigma"),
                int(inner != "scale"),
                int(inner != "scale"),
            ],
            "parameterType": [
                None,
                None,
                InnerParameterType.SCALING if inner != "sigma" else None,
                InnerParameterType.SCALING if inner != "sigma" else None,
                InnerParameterType.SIGMA if inner != "scale" else None,
                InnerParameterType.SIGMA if inner != "scale" else None,
            ],
        }
    )
    return petab.Problem(
        model=model,
        observable_df=petab.get_observable_df(observable),
        parameter_df=petab.get_parameter_df(parameter),
        condition_df=petab.get_condition_df(
            pd.DataFrame({petab.CONDITION_ID: ["c1", "c2"], "A": [1.0, 2.0]})
        ),
        measurement_df=petab.get_measurement_df(pd.DataFrame(rows)),
    )


@pytest.fixture(scope="module")
def model_folders(tmp_path_factory):
    return {
        base: tmp_path_factory.mktemp(f"lognormal_{base}")
        for base in [petab.LOG, petab.LOG10]
    }


def create_imported_problem(problem, folder, hierarchical):
    return PetabImporter(
        problem,
        hierarchical=hierarchical,
        model_name=f"native_lognormal_{folder.name}",
        output_folder=str(folder),
    ).create_problem()


def record_case(data):
    """Optionally retain exact comparisons outside the test repository."""
    if filename := os.environ.get("LOGNORMAL_EVIDENCE"):
        with Path(filename).open("a") as stream:
            stream.write(json.dumps(data) + "\n")


@pytest.mark.parametrize("base", [petab.LOG, petab.LOG10])
@pytest.mark.parametrize("inner", ["scale", "sigma", "both"])
@pytest.mark.parametrize(
    "method", [SensitivityMethod.forward, SensitivityMethod.adjoint]
)
def test_native_lognormal_against_amici(base, inner, method, model_folders):
    problem = make_problem(base, inner)
    check_pipeline(
        problem, model_folders[base], method, base=base, inner=inner
    )


def check_pipeline(problem, folder, method, **label):
    hierarchical_problem = create_imported_problem(problem, folder, True)
    full_problem = create_imported_problem(problem, folder, False)
    objective = hierarchical_problem.objective
    objective_full = full_problem.objective
    names = [
        hierarchical_problem.x_names[i]
        for i in hierarchical_problem.x_free_indices
    ]
    full_names = [full_problem.x_names[i] for i in full_problem.x_free_indices]
    for obj in [objective, objective_full]:
        obj.amici_solver.set_absolute_tolerance(1e-12)
        obj.amici_solver.set_relative_tolerance(1e-12)
    objective.amici_solver.set_sensitivity_method(method)
    point = {"k1": np.log10(0.4), "k2": 0.15}
    x = np.array([point[name] for name in names])
    result = objective(x, sensi_orders=(0, 1), return_dict=True)
    inner_ids = objective.calculator.get_inner_par_ids()
    point.update(zip(inner_ids, result[INNER_PARAMETERS], strict=True))
    full = objective_full(
        np.array([point[name] for name in full_names]),
        sensi_orders=(0, 1),
        return_dict=True,
    )
    for profiled_rdata, standard_rdata in zip(
        result["rdatas"], full["rdatas"], strict=True
    ):
        np.testing.assert_allclose(
            profiled_rdata["y"], standard_rdata["y"], rtol=1e-7, atol=1e-12
        )
        np.testing.assert_allclose(
            profiled_rdata["sigmay"],
            standard_rdata["sigmay"],
            rtol=1e-7,
            atol=1e-12,
        )
    expected_gradient = np.array(
        [full[GRAD][full_names.index(name)] for name in names]
    )
    f_error = float(abs(result[FVAL] - full[FVAL]))
    gradient_error = float(
        np.max(
            np.abs(result[GRAD] - expected_gradient)
            / (1 + np.abs(expected_gradient))
        )
    )
    assert f_error / (1 + abs(full[FVAL])) < 1e-6
    assert gradient_error < 1e-5
    step = 1e-5
    fd = np.array(
        [
            (
                objective(x + np.eye(len(x))[i] * step)
                - objective(x - np.eye(len(x))[i] * step)
            )
            / (2 * step)
            for i in range(len(x))
        ]
    )
    fd_error = float(np.max(np.abs(result[GRAD] - fd) / (1 + np.abs(fd))))
    assert fd_error < 1e-5
    if label.get("bounded"):
        assert np.isclose(point["s1"], 0.4)
    record_case(
        {
            **label,
            "method": str(method),
            "nllh": float(result[FVAL]),
            "amici_nllh": float(full[FVAL]),
            "nllh_error": f_error,
            "amici_gradient_error": gradient_error,
            "finite_difference_error": fd_error,
            "inner_parameters_scaled": {
                name: float(point[name]) for name in inner_ids
            },
        }
    )
    return objective, x


@pytest.mark.parametrize("base", [petab.LOG, petab.LOG10])
def test_bounded_lognormal_scale(base, model_folders):
    check_pipeline(
        make_problem(base, bounded=True),
        model_folders[base],
        SensitivityMethod.forward,
        base=base,
        bounded=True,
    )


@pytest.mark.parametrize("case", ["offset", "additive_term", "prior"])
def test_lognormal_unsupported_formulae(case):
    problem = make_problem(petab.LOG)
    if case == "offset":
        problem.observable_df.loc["relative", petab.OBSERVABLE_FORMULA] += "+b"
        problem.parameter_df.loc["b"] = problem.parameter_df.loc["s1"].copy()
        problem.parameter_df.loc["b", "parameterType"] = (
            InnerParameterType.OFFSET
        )
        message = "offset"
    elif case == "additive_term":
        problem.observable_df.loc["relative", petab.OBSERVABLE_FORMULA] += "+1"
        message = "multiply the entire"
    else:
        problem.parameter_df[petab.OBJECTIVE_PRIOR_TYPE] = None
        problem.parameter_df.loc["s1", petab.OBJECTIVE_PRIOR_TYPE] = "normal"
        problem.parameter_df.loc["s1", petab.OBJECTIVE_PRIOR_PARAMETERS] = (
            "1;0.2"
        )
        message = "priors"
    with pytest.raises(NotImplementedError, match=message):
        validate_hierarchical_petab_problem(problem)


@pytest.mark.parametrize("case", ["mixed", "partial_sigma", "zero"])
def test_lognormal_unsupported_inputs(case, model_folders):
    problem = make_problem(petab.LOG)
    if case == "mixed":
        problem.parameter_df.loc["qx"] = problem.parameter_df.loc["q1"].copy()
        mask = problem.measurement_df[petab.OBSERVABLE_ID] == "quantitative"
        problem.measurement_df.loc[mask, petab.NOISE_PARAMETERS] = "qx"
        message = "Mixed linear/log"
        error = NotImplementedError
    elif case == "partial_sigma":
        relative = problem.measurement_df[petab.OBSERVABLE_ID] == "relative"
        problem.measurement_df.loc[relative, petab.OBSERVABLE_PARAMETERS] = (
            "s1"
        )
        for parameter in ["s2", "q2"]:
            problem.parameter_df.loc[parameter, petab.ESTIMATE] = 0
            problem.parameter_df.loc[parameter, "parameterType"] = None
        problem.parameter_df = problem.parameter_df.drop(index="s2")
        message = "Every measurement"
        error = NotImplementedError
    else:
        problem.measurement_df.loc[0, petab.MEASUREMENT] = 0.0
        message = "positive"
        error = ValueError
    with pytest.raises(error, match=message):
        create_imported_problem(problem, model_folders[petab.LOG], True)


def test_lognormal_degenerate_sigma(model_folders):
    problem = make_problem(petab.LOG, "sigma")
    # At t=0, A(0)=1 in c1; fixed scaling s1=1. Both observations equal
    # their prediction, so sigma has no finite positive MLE.
    selected = problem.measurement_df[
        problem.measurement_df[petab.TIME] == 0.0
    ]
    problem.measurement_df = pd.concat([selected, selected], ignore_index=True)
    relative = problem.measurement_df[petab.OBSERVABLE_ID] == "relative"
    problem.measurement_df.loc[relative, petab.MEASUREMENT] = (
        problem.measurement_df.loc[
            relative, petab.SIMULATION_CONDITION_ID
        ].map({"c1": 1.0, "c2": 2.0})
    )
    objective = create_imported_problem(
        problem, model_folders[petab.LOG], True
    ).objective
    with pytest.raises(ValueError, match="No observations|No finite positive"):
        objective(np.array([np.log10(0.4), 0.15]))


@pytest.mark.parametrize(
    "method", [SensitivityMethod.forward, SensitivityMethod.adjoint]
)
def test_shared_log_sigma(method, model_folders):
    problem = make_problem(petab.LOG)
    relative = problem.measurement_df[petab.OBSERVABLE_ID] == "relative"
    problem.measurement_df.loc[relative, petab.NOISE_PARAMETERS] = "q1"
    problem.parameter_df = problem.parameter_df.drop(index="q2")
    check_pipeline(
        problem,
        model_folders[petab.LOG],
        method,
        base=petab.LOG,
        shared_sigma=True,
    )


@pytest.mark.parametrize("mode,orders", [(MODE_RES, (0,)), (MODE_FUN, (0, 2))])
def test_lognormal_unsupported_modes(mode, orders, model_folders):
    objective = create_imported_problem(
        make_problem(petab.LOG), model_folders[petab.LOG], True
    ).objective
    with pytest.raises(
        NotImplementedError, match="function values and first derivatives"
    ):
        objective(
            np.array([np.log10(0.4), 0.15]), sensi_orders=orders, mode=mode
        )


def test_native_lognormal_optimization(model_folders):
    problem = create_imported_problem(
        make_problem(petab.LOG), model_folders[petab.LOG], True
    )
    problem.objective.amici_solver.set_relative_tolerance(1e-12)
    problem.objective.amici_solver.set_absolute_tolerance(1e-12)
    problem.objective.amici_solver.set_relative_tolerance_fsa(1e-12)
    problem.objective.amici_solver.set_absolute_tolerance_fsa(1e-12)
    problem.set_x_guesses(np.array([[np.log10(0.4), 0.15]]))
    start = float(problem.objective(problem.x_guesses[0]))
    result = minimize(
        problem,
        optimizer=ScipyOptimizer(
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6},
        ),
        n_starts=1,
        progress_bar=False,
    ).optimize_result.list[0]
    assert result.exitflag == 0
    assert np.isfinite(result.fval) and result.fval < start
    record_case(
        {
            "optimization": True,
            "initial_nllh": start,
            "final_nllh": float(result.fval),
            "exitflag": int(result.exitflag),
        }
    )
