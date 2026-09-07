"""Tests for the PEtab v2 sensitivity-to-parameter index slices."""

import numpy as np
import petab.v2
import pytest
from benchmark_models_petab import (
    MODELS_DIR,  # noqa: F401
    get_problem_yaml_path,
)

from pypesto.C import RDATAS
from pypesto.objective.amici.amici_util import (
    petab_v2_index_slices,
    petab_v2_placeholder_mapping,
)
from pypesto.petab import PetabImporter


@pytest.fixture(scope="module")
def boehm_v2_objective():
    """A plain (non-hierarchical) PEtab v2 objective and its simulator."""
    petab_problem = petab.v2.Problem.from_yaml(
        get_problem_yaml_path("Boehm_JProteomeRes2014")
    )
    objective = (
        PetabImporter(petab_problem, model_name="Boehm_v2_index_slices")
        .create_objective_creator()
        .create_objective()
    )
    return petab_problem, objective


def test_petab_v2_index_slices_match_amici_gradient(boehm_v2_objective):
    """The slices map per-experiment sensitivities onto the right parameters.

    AMICI aggregates the per-experiment sensitivities into a gradient itself.
    Doing the same aggregation through the index slices must reproduce it, so
    this pins the sensitivity-to-parameter correspondence end to end.
    """
    petab_problem, objective = boehm_v2_objective
    simulator = objective.calculator.petab_simulator

    x_nominal = petab_problem.get_x_nominal_dict()
    x = np.asarray([x_nominal[x_id] for x_id in objective.x_ids])
    ret = objective(x, sensi_orders=(0, 1), return_dict=True)
    rdatas = ret[RDATAS]

    index_slices = petab_v2_index_slices(
        petab_problem=simulator.exp_man.petab_problem,
        par_sim_ids=simulator.model.get_free_parameter_ids(),
        edatas=objective.edatas,
        par_opt_ids=objective.x_ids,
    )
    assert len(index_slices) == len(rdatas)

    # `rdata.sllh` is the log-likelihood gradient in `plist` order, per
    #  experiment; the objective returns the negative log-likelihood gradient
    assembled = np.zeros(len(objective.x_ids))
    for rdata, (par_sim_slice, par_opt_slice) in zip(
        rdatas, index_slices, strict=True
    ):
        np.add.at(assembled, par_opt_slice, -rdata.sllh[par_sim_slice])

    assert np.allclose(assembled, ret["grad"], rtol=1e-8, atol=1e-8), (
        f"assembled={assembled}\nexpected={ret['grad']}"
    )
    # the test is only meaningful if the slices actually select something
    assert sum(len(s) for s, _ in index_slices) > 0


def test_petab_v2_index_slices_omit_unknown_parameters(boehm_v2_objective):
    """Parameters outside ``par_opt_ids`` are omitted from the slices.

    This is what lets the hierarchical path drop the parameters it solves for
    analytically from the gradient assembly.
    """
    petab_problem, objective = boehm_v2_objective
    simulator = objective.calculator.petab_simulator

    kwargs = {
        "petab_problem": simulator.exp_man.petab_problem,
        "par_sim_ids": simulator.model.get_free_parameter_ids(),
        "edatas": objective.edatas,
    }
    full = petab_v2_index_slices(par_opt_ids=objective.x_ids, **kwargs)
    dropped = petab_v2_index_slices(par_opt_ids=objective.x_ids[1:], **kwargs)

    n_full = sum(len(s) for s, _ in full)
    n_dropped = sum(len(s) for s, _ in dropped)
    assert n_dropped < n_full
    # ... and the surviving entries still point at the right parameters
    for (_, opt_full), (sim_d, opt_d) in zip(full, dropped, strict=True):
        assert len(sim_d) == len(opt_d)
        # every remaining optimization index is a valid position in the
        #  shortened id list
        assert all(0 <= ix < len(objective.x_ids) - 1 for ix in opt_d)
        assert len(opt_d) <= len(opt_full)


def _problem_with_one_placeholder(objective):
    """Deep copy of the problem, with one observable given a placeholder.

    Returns the copied problem, its first experiment, and two measurements of
    the same observable within that experiment. Only used for
    `petab_v2_placeholder_mapping`, which reads the tables and never
    simulates, so the problem need not stay consistent with the model.
    """
    import copy

    petab_problem = copy.deepcopy(
        objective.calculator.petab_simulator.exp_man.petab_problem
    )
    experiment = petab_problem.experiments[0]
    measurements = petab_problem.get_measurements_for_experiment(experiment)

    by_observable = {}
    for measurement in measurements:
        by_observable.setdefault(measurement.observable_id, []).append(
            measurement
        )
    observable_id, siblings = next(
        (obs_id, ms) for obs_id, ms in by_observable.items() if len(ms) >= 2
    )
    observable = next(
        o for o in petab_problem.observables if o.id == observable_id
    )
    observable.observable_placeholders = [
        f"observableParameter1_{observable_id}"
    ]
    return petab_problem, experiment, siblings


def test_petab_v2_placeholder_mapping_resolves_overrides(boehm_v2_objective):
    """A placeholder overridden consistently maps to the overriding id."""
    _, objective = boehm_v2_objective
    petab_problem, experiment, siblings = _problem_with_one_placeholder(
        objective
    )
    observable_id = siblings[0].observable_id
    for measurement in siblings:
        measurement.observable_parameters = ["some_parameter"]

    mapping = petab_v2_placeholder_mapping(petab_problem, experiment)
    assert mapping[f"observableParameter1_{observable_id}"] == "some_parameter"


def test_petab_v2_placeholder_mapping_ignores_numeric_overrides(
    boehm_v2_objective,
):
    """A numeric override is not a parameter and is left out of the mapping."""
    _, objective = boehm_v2_objective
    petab_problem, experiment, siblings = _problem_with_one_placeholder(
        objective
    )
    observable_id = siblings[0].observable_id
    for measurement in siblings:
        measurement.observable_parameters = [1.0]

    # other observables of the experiment may contribute their own
    #  placeholders, so only assert about the one under test
    mapping = petab_v2_placeholder_mapping(petab_problem, experiment)
    assert f"observableParameter1_{observable_id}" not in mapping
