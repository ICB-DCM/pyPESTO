"""Inner optimization problem in hierarchical optimization."""

import logging

import pandas as pd

from ...C import (
    MEASUREMENT_TYPE,
    PARAMETER_TYPE,
    SEMIQUANTITATIVE,
    InnerParameterType,
)
from ..base_problem import (
    AmiciInnerProblem,
    _get_timepoints_with_replicates,
    ix_matrices_from_arrays,
)
from .parameter import RelativeInnerParameter

try:
    import amici
    import petab.v1 as petab
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
        amici_model: "amici.Model",
        edatas: list["amici.ExpData"],
    ) -> "RelativeInnerProblem":
        """Create an InnerProblem from a PEtab problem and AMICI objects."""
        return inner_problem_from_petab_problem(
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
    amici_model: "amici.Model",
    edatas: list["amici.ExpData"],
) -> RelativeInnerProblem:
    """
    Create inner problem from PEtab problem.

    Hierarchical optimization is a pypesto-specific PEtab extension.
    """
    import amici

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
    data = [amici.numpy.ExpDataView(edata)["observedData"] for edata in edatas]

    # matrixify
    ix_matrices = ix_matrices_from_arrays(ixs, data)

    # assign matrices, observable indices and ids to inner parameters
    for par in inner_parameters:
        par.ixs = ix_matrices[par.inner_parameter_id]
        par.observable_indices = [
            meas_indices[2] for meas_indices in ixs[par.inner_parameter_id]
        ]
        par.observable_ids = [
            amici_model.getObservableIds()[obs_idx]
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
    amici_model: "amici.Model",
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
    observable_ids = amici_model.getObservableIds()

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
