"""Inner optimization problem in hierarchical optimization."""

import logging
import warnings
from collections import Counter

import pandas as pd

from ...C import (
    INNER_PARAMETER_BOUNDS,
    LIN,
    MEASUREMENT_TYPE,
    PARAMETER_TYPE,
    SEMIQUANTITATIVE,
    InnerParameterType,
)
from ...C import LOWER_BOUND as PYPESTO_LOWER_BOUND
from ...C import UPPER_BOUND as PYPESTO_UPPER_BOUND
from ..base_problem import (
    AmiciInnerProblem,
    _get_timepoints_with_replicates,
    ix_matrices_from_arrays,
)
from .parameter import RelativeInnerParameter

try:
    import amici.sim.sundials as asd
    import petab.v1 as petab
    from petab import v2
    from petab.v1.C import (
        ESTIMATE,
        LOWER_BOUND,
        NOISE_PARAMETERS,
        OBSERVABLE_ID,
        OBSERVABLE_PARAMETERS,
        PARAMETER_ID,
        PARAMETER_SCALE,
        TIME,
        UPPER_BOUND,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)


class RelativeInnerProblem(AmiciInnerProblem):
    r"""Inner optimization problem for relative data with scaling/offset.

    Attributes
    ----------
    xs:
        Mapping of (inner) parameter ID to ``InnerParameters``.
    data:
        Measurement data. One matrix (`num_timepoints` x `num_observables`)
        per simulation condition. Missing observations as NaN.
    edatas:
        AMICI ``ExpData``\s for each simulation condition.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_petab_amici(
        petab_problem: "petab.Problem",
        amici_model: "asd.Model",
        edatas: list["asd.ExpData"],
    ) -> "RelativeInnerProblem":
        """Create an InnerProblem from a PEtab problem and AMICI objects."""
        return inner_problem_from_petab_problem(
            petab_problem, amici_model, edatas
        )

    @staticmethod
    def from_petab_v2_amici(
        petab_problem: "v2.Problem",
        amici_model: "asd.Model",
        edatas: list["asd.ExpData"],
    ) -> "RelativeInnerProblem":
        """Create an InnerProblem from a PEtab v2 problem and AMICI objects.

        Parameters
        ----------
        petab_problem:
            The PEtab v2 problem, as used by the
            :class:`amici.sim.sundials.petab.ExperimentManager` that created
            ``edatas`` (i.e., the problem preprocessed by the AMICI PEtab
            importer).
        amici_model:
            The AMICI model.
        edatas:
            The experimental data, one per PEtab experiment, in the order of
            ``petab_problem.experiments``.
        """
        return inner_problem_from_petab_v2_problem(
            petab_problem, amici_model, edatas
        )

    def get_relative_observable_ids(self) -> list[str]:
        """Get IDs of all unique relative observables with scaling and/or offset."""
        return list(
            {
                observable_id
                for x in self.xs.values()
                if x.inner_parameter_type
                in [
                    InnerParameterType.SCALING,
                    InnerParameterType.OFFSET,
                ]
                for observable_id in x.observable_ids
            }
        )

    def get_observable_indices_for_xs(
        self, inner_parameter_type: str
    ) -> list[int]:
        """Get unique list of ``RelativeParameter.observable_indices`` values."""
        return list(
            {
                obs_idx
                for x in self.xs.values()
                if x.inner_parameter_type == inner_parameter_type
                for obs_idx in x.observable_indices
            }
        )

    def get_xs_for_obs_idx(self, obs_idx: int) -> list[RelativeInnerParameter]:
        r"""Get ``RelativeParameter``\s that belong to the observable with index `obs_idx`."""
        return [x for x in self.xs.values() if obs_idx in x.observable_indices]


def inner_problem_from_petab_problem(
    petab_problem: "petab.Problem",
    amici_model: "asd.Model",
    edatas: list["asd.ExpData"],
) -> RelativeInnerProblem:
    """
    Create inner problem from PEtab problem.

    Hierarchical optimization is a pypesto-specific PEtab extension.
    """

    # inner parameters
    inner_parameters = inner_parameters_from_parameter_df(
        petab_problem.parameter_df, petab_problem.measurement_df
    )

    x_ids = [x.inner_parameter_id for x in inner_parameters]

    # used indices for all measurement specific parameters
    ixs = ixs_for_measurement_specific_parameters(
        petab_problem, amici_model, x_ids
    )

    # transform experimental data
    data = [asd.ExpDataView(edata)["measurements"] for edata in edatas]

    # matrixify
    ix_matrices = ix_matrices_from_arrays(ixs, data)

    # assign matrices, observable indices and ids to inner parameters
    for par in inner_parameters:
        par.ixs = ix_matrices[par.inner_parameter_id]
        par.observable_indices = [
            meas_indices[2] for meas_indices in ixs[par.inner_parameter_id]
        ]
        par.observable_ids = [
            amici_model.get_observable_ids()[obs_idx]
            for obs_idx in par.observable_indices
        ]

    par_group_types = {
        tuple(obs_pars.split(";")): (
            petab_problem.parameter_df.loc[obs_par, PARAMETER_TYPE]
            for obs_par in obs_pars.split(";")
        )
        for (obs_id, obs_pars), _ in petab_problem.measurement_df.groupby(
            [petab.OBSERVABLE_ID, petab.OBSERVABLE_PARAMETERS], dropna=True
        )
        if ";" in obs_pars  # prefilter for at least 2 observable parameters
    }

    coupled_pars = {
        group
        for group, types in par_group_types.items()
        if (
            (InnerParameterType.SCALING in types)
            and (InnerParameterType.OFFSET in types)
        )
    }

    # Check each group is of length 2
    for group in coupled_pars:
        if len(group) != 2:
            raise ValueError(
                f"Expected exactly 2 parameters in group {group}: a scaling "
                f"and an offset parameter."
            )

    id_to_par = {par.inner_parameter_id: par for par in inner_parameters}

    # assign coupling
    for par in inner_parameters:
        if par.inner_parameter_type not in [
            InnerParameterType.SCALING,
            InnerParameterType.OFFSET,
        ]:
            continue
        for group in coupled_pars:
            if par.inner_parameter_id in group:
                coupled_parameter_id = group[
                    group.index(par.inner_parameter_id) - 1
                ]
                par.coupled = id_to_par[coupled_parameter_id]
                break

    return RelativeInnerProblem(xs=inner_parameters, data=data, edatas=edatas)


def inner_parameters_from_parameter_df(
    par_df: pd.DataFrame,
    meas_df: pd.DataFrame,
) -> list[RelativeInnerParameter]:
    """
    Create list of inner free parameters from PEtab parameter table.

    Inner parameters are those that have a non-empty `parameterType` in the
    PEtab problem.
    """
    # create list of hierarchical parameters
    par_df = par_df.reset_index()

    for col in (PARAMETER_TYPE,):
        if col not in par_df:
            par_df[col] = None

    parameters = []

    for _, row in par_df.iterrows():
        if not row[ESTIMATE]:
            continue
        if petab.is_empty(row[PARAMETER_TYPE]):
            continue
        # If a sigma parameter belongs to a semiquantitative
        # observable, it is not a relative inner parameter.
        if row[PARAMETER_TYPE] == InnerParameterType.SIGMA:
            if MEASUREMENT_TYPE in meas_df.columns:
                par_id = row[PARAMETER_ID]
                corresponding_measurements = meas_df[
                    meas_df[NOISE_PARAMETERS] == par_id
                ]
                if any(
                    corresponding_measurements[MEASUREMENT_TYPE]
                    == SEMIQUANTITATIVE
                ):
                    continue

        parameters.append(
            RelativeInnerParameter(
                inner_parameter_id=row[PARAMETER_ID],
                inner_parameter_type=row[PARAMETER_TYPE],
                scale=row[PARAMETER_SCALE],
                lb=row[LOWER_BOUND],
                ub=row[UPPER_BOUND],
                observable_ids=None,
                observable_indices=None,
            )
        )

    return parameters


def ixs_for_measurement_specific_parameters(
    petab_problem: "petab.Problem",
    amici_model: "asd.Model",
    x_ids: list[str],
) -> dict[str, list[tuple[int, int, int]]]:
    """
    Create mapping of parameters to measurements.

    Returns
    -------
    A dictionary mapping parameter ID to a list of
    `(condition index, time index, observable index)` tuples in which this
    output parameter is used. For each condition, the time index refers to
    a sorted list of non-unique time points for which there are measurements.
    """
    ixs_for_par = {}
    observable_ids = amici_model.get_observable_ids()

    simulation_conditions = (
        petab_problem.get_simulation_conditions_from_measurement_df()
    )
    for condition_ix, condition in simulation_conditions.iterrows():
        # measurement table for current condition
        df_for_condition = petab.get_rows_for_condition(
            measurement_df=petab_problem.measurement_df, condition=condition
        )

        # unique sorted list of timepoints
        timepoints = sorted(df_for_condition[TIME].unique().astype(float))
        # non-unique sorted list of timepoints
        timepoints_w_reps = _get_timepoints_with_replicates(
            measurement_df=df_for_condition
        )

        for time in timepoints:
            # subselect measurements for time `time`
            df_for_time = df_for_condition[df_for_condition[TIME] == time]
            time_ix_0 = timepoints_w_reps.index(time)

            # remember used time indices for each observable
            time_ix_for_obs_ix = {}

            # iterate over measurements
            for _, measurement in df_for_time.iterrows():
                # extract observable index
                observable_ix = observable_ids.index(
                    measurement[OBSERVABLE_ID]
                )

                # as the time indices have to account for replicates, we need
                #  to track which time indices have already been assigned for
                #  the current observable
                if observable_ix in time_ix_for_obs_ix:
                    # a replicate
                    time_ix_for_obs_ix[observable_ix] += 1
                else:
                    # the first measurement for this `(observable, timepoint)`
                    time_ix_for_obs_ix[observable_ix] = time_ix_0
                time_w_reps_ix = time_ix_for_obs_ix[observable_ix]

                observable_overrides = petab.split_parameter_replacement_list(
                    measurement.get(OBSERVABLE_PARAMETERS, None)
                )
                noise_overrides = petab.split_parameter_replacement_list(
                    measurement.get(NOISE_PARAMETERS, None)
                )

                # try to insert if hierarchical parameter
                for override in observable_overrides + noise_overrides:
                    if override in x_ids:
                        ixs_for_par.setdefault(override, []).append(
                            (condition_ix, time_w_reps_ix, observable_ix)
                        )
    return ixs_for_par


def inner_problem_from_petab_v2_problem(
    petab_problem: "v2.Problem",
    amici_model: "asd.Model",
    edatas: list["asd.ExpData"],
) -> RelativeInnerProblem:
    """
    Create inner problem from a PEtab v2 problem.

    Hierarchical optimization is a pypesto-specific PEtab extension.

    See :meth:`RelativeInnerProblem.from_petab_v2_amici` for the expected
    arguments.
    """
    from ...petab.util import get_petab_v2_extra_field

    # inner parameters
    inner_parameters = inner_parameters_from_petab_v2_problem(petab_problem)

    x_ids = [x.inner_parameter_id for x in inner_parameters]

    # used indices for all measurement specific parameters
    ixs = ixs_for_measurement_specific_parameters_v2(
        petab_problem, amici_model, x_ids
    )

    # transform experimental data
    data = [asd.ExpDataView(edata)["measurements"] for edata in edatas]

    # matrixify
    ix_matrices = ix_matrices_from_arrays(ixs, data)

    # inner parameters are located via the measurement-specific overrides; one
    #  that only occurs directly in a formula cannot be assigned to
    #  measurements (see `ixs_for_measurement_specific_parameters_v2`)
    if missing := [x for x in x_ids if x not in ixs]:
        raise NotImplementedError(
            "The following inner parameters are not used by any measurement: "
            f"{sorted(missing)}. Either they override no observable/noise "
            "placeholder -- inner parameters referenced directly in a formula "
            "are not supported, use a placeholder overridden per measurement "
            "instead -- or the observable they belong to has no measurements "
            "in any experiment."
        )

    # assign matrices, observable indices and ids to inner parameters
    for par in inner_parameters:
        par.ixs = ix_matrices[par.inner_parameter_id]
        par.observable_indices = [
            meas_indices[2] for meas_indices in ixs[par.inner_parameter_id]
        ]
        par.observable_ids = [
            amici_model.get_observable_ids()[obs_idx]
            for obs_idx in par.observable_indices
        ]

    # detect coupled scaling and offset parameters, i.e., pairs of scaling
    #  and offset parameters that override the placeholders of the same
    #  measurement
    # scaling/offset annotations of *all* parameters, with a flag for whether
    #  they are estimated. Non-estimated ones are not inner parameters, but a
    #  scaling whose offset partner is not estimated cannot be solved for
    #  analytically and must be rejected rather than silently left uncoupled.
    annotated_types = {}
    for parameter in petab_problem.parameters:
        parameter_type = get_petab_v2_extra_field(parameter, PARAMETER_TYPE)
        if parameter_type in (
            InnerParameterType.SCALING,
            InnerParameterType.OFFSET,
        ):
            annotated_types[parameter.id] = (
                InnerParameterType(parameter_type),
                bool(parameter.estimate),
            )

    coupled_pars = set()
    for measurement in petab_problem.measurements:
        # only scaling/offset overrides can form a coupled pair; numeric and
        #  unrelated overrides must not count towards the group
        group = tuple(
            override
            for override in map(str, measurement.observable_parameters)
            if override in annotated_types
        )
        # prefilter for at least 2 observable parameters
        if len(group) < 2:
            continue
        types = [annotated_types[override][0] for override in group]
        if (InnerParameterType.SCALING in types) and (
            InnerParameterType.OFFSET in types
        ):
            not_estimated = [
                override
                for override in group
                if not annotated_types[override][1]
            ]
            if len(not_estimated) == len(group):
                # none of them is an inner parameter: this observable is not
                #  hierarchically optimized, so there is nothing to couple
                continue
            if not_estimated:
                # a partially estimated pair cannot be solved for
                #  analytically, and must not be silently left uncoupled
                raise NotImplementedError(
                    f"Observable parameters {list(group)} form a coupled "
                    f"scaling/offset pair, but {sorted(not_estimated)} are not "
                    "estimated. Coupled scaling/offset parameters must either "
                    "all be estimated or all be non-estimated."
                )
            coupled_pars.add(group)

    # Check each group is of length 2
    for group in coupled_pars:
        if len(group) != 2:
            raise ValueError(
                f"Expected exactly 2 parameters in group {group}: a scaling "
                f"and an offset parameter."
            )

    id_to_par = {par.inner_parameter_id: par for par in inner_parameters}

    # assign coupling
    for par in inner_parameters:
        if par.inner_parameter_type not in [
            InnerParameterType.SCALING,
            InnerParameterType.OFFSET,
        ]:
            continue
        for group in coupled_pars:
            if par.inner_parameter_id in group:
                coupled_parameter_id = group[
                    group.index(par.inner_parameter_id) - 1
                ]
                par.coupled = id_to_par[coupled_parameter_id]
                break

    return RelativeInnerProblem(xs=inner_parameters, data=data, edatas=edatas)


def inner_parameters_from_petab_v2_problem(
    petab_problem: "v2.Problem",
) -> list[RelativeInnerParameter]:
    """
    Create list of inner free parameters from a PEtab v2 problem.

    Inner parameters are those that have a non-empty `parameterType` extra
    field (column) in the PEtab parameter table.
    """
    from ...petab.util import get_petab_v2_extra_field

    parameters = []
    # collected so that a single warning is emitted rather than one per
    #  parameter (whose differing messages the `warnings` filter cannot fold)
    overridden_bounds = []

    for parameter in petab_problem.parameters:
        if not parameter.estimate:
            continue
        parameter_type = get_petab_v2_extra_field(parameter, PARAMETER_TYPE)
        if parameter_type is None:
            continue
        # Note: sigma parameters of semiquantitative observables are not
        #  relative inner parameters -- irrelevant here, as non-quantitative
        #  data types other than relative are not yet supported for PEtab v2
        #  (see `validate_hierarchical_petab_problem_v2`).

        inner_parameter_type = InnerParameterType(parameter_type)
        # Scaling and offset parameters can be bounded arbitrarily; all other
        #  inner parameter types (in particular sigma) require the fixed
        #  bounds of `INNER_PARAMETER_BOUNDS`. This is the PEtab v2 equivalent
        #  of `correct_parameter_df_bounds`, which only works on DataFrames.
        lb, ub = parameter.lb, parameter.ub
        if inner_parameter_type not in (
            InnerParameterType.SCALING,
            InnerParameterType.OFFSET,
        ):
            bounds = INNER_PARAMETER_BOUNDS[inner_parameter_type]
            lb = bounds[PYPESTO_LOWER_BOUND]
            ub = bounds[PYPESTO_UPPER_BOUND]
            if (parameter.lb, parameter.ub) != (lb, ub):
                overridden_bounds.append(
                    f"`{parameter.id}` [{parameter.lb}, {parameter.ub}]"
                )

        parameters.append(
            RelativeInnerParameter(
                inner_parameter_id=parameter.id,
                inner_parameter_type=inner_parameter_type,
                # PEtab v2 does not support parameter scales
                scale=LIN,
                lb=lb,
                ub=ub,
                observable_ids=None,
                observable_indices=None,
            )
        )

    if overridden_bounds:
        # `stacklevel` is deliberately left at the `warn` call: the number of
        #  frames up to the public entry point is not fixed, so any other
        #  value would point at an arbitrary internal frame. The message names
        #  the affected parameters instead.
        warnings.warn(
            "Ignoring the bounds declared for the following inner "
            f"parameters: {', '.join(overridden_bounds)}. Hierarchical "
            "optimization requires the fixed bounds of "
            "`INNER_PARAMETER_BOUNDS` for all inner parameter types except "
            "scaling and offset.",
            stacklevel=1,
        )

    return parameters


def ixs_for_measurement_specific_parameters_v2(
    petab_problem: "v2.Problem",
    amici_model: "asd.Model",
    x_ids: list[str],
) -> dict[str, list[tuple[int, int, int]]]:
    """
    Create mapping of parameters to measurements for a PEtab v2 problem.

    The layout of the returned indices matches the ``ExpData`` objects
    created by :class:`amici.sim.sundials.petab.ExperimentManager` for the
    given problem: the condition index refers to the position of the
    experiment in ``petab_problem.experiments``, and the time index refers to
    a sorted list of non-unique time points for which there are measurements
    in the respective experiment.

    Returns
    -------
    A dictionary mapping parameter ID to a list of
    `(condition index, time index, observable index)` tuples in which this
    output parameter is used.
    """
    ixs_for_par = {}
    observable_ids = amici_model.get_observable_ids()

    for condition_ix, experiment in enumerate(petab_problem.experiments):
        measurements = petab_problem.get_measurements_for_experiment(
            experiment
        )

        # non-unique sorted list of timepoints: the superset of the
        #  timepoints of the measurements of all observables, including
        #  replicates (see `ExperimentManager._set_timepoints_and_measurements`)
        t_counters: dict[str, Counter] = {}
        for measurement in measurements:
            t_counters.setdefault(measurement.observable_id, Counter()).update(
                [measurement.time]
            )
        max_counter = Counter()
        for counter in t_counters.values():
            for time, count in counter.items():
                max_counter[time] = max(max_counter[time], count)
        timepoints_w_reps = sorted(max_counter.elements())

        time_to_meas = {}
        for measurement in measurements:
            time_to_meas.setdefault(measurement.time, []).append(measurement)

        for time in sorted(time_to_meas):
            time_ix_0 = timepoints_w_reps.index(time)

            # remember used time indices for each observable
            time_ix_for_obs_ix = {}

            # iterate over measurements
            for measurement in time_to_meas[time]:
                # extract observable index
                observable_ix = observable_ids.index(measurement.observable_id)

                # as the time indices have to account for replicates, we need
                #  to track which time indices have already been assigned for
                #  the current observable
                if observable_ix in time_ix_for_obs_ix:
                    # a replicate
                    time_ix_for_obs_ix[observable_ix] += 1
                else:
                    # the first measurement for this `(observable, timepoint)`
                    time_ix_for_obs_ix[observable_ix] = time_ix_0
                time_w_reps_ix = time_ix_for_obs_ix[observable_ix]

                overrides = [
                    str(override)
                    for override in (
                        list(measurement.observable_parameters)
                        + list(measurement.noise_parameters)
                    )
                ]

                # try to insert if hierarchical parameter
                for override in overrides:
                    if override in x_ids:
                        ixs_for_par.setdefault(override, []).append(
                            (condition_ix, time_w_reps_ix, observable_ix)
                        )
    return ixs_for_par
