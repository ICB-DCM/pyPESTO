"""Inner optimization problem in hierarchical optimization."""

import logging

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
    from petab.v2.C import EXPERIMENT_ID
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
        _ixs_for_condition(
            df_for_condition, condition_ix, observable_ids, x_ids, ixs_for_par
        )
    return ixs_for_par


def _ixs_for_condition(
    df_for_condition: pd.DataFrame,
    condition_ix: int,
    observable_ids: list[str],
    x_ids: list[str],
    ixs_for_par: dict[str, list[tuple[int, int, int]]],
) -> None:
    """Add the measurement indices of one condition to ``ixs_for_par``.

    See :func:`ixs_for_measurement_specific_parameters`.
    """
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
            observable_ix = observable_ids.index(measurement[OBSERVABLE_ID])

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


def inner_problem_from_petab_v2_problem(
    petab_problem: "v2.Problem",
    amici_model: "asd.Model",
    edatas: list["asd.ExpData"],
) -> RelativeInnerProblem:
    """
    Create inner problem from a PEtab v2 problem.

    See :meth:`RelativeInnerProblem.from_petab_v2_amici`.
    """
    from ..petab import get_inner_parameters_v2

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
    #  measurement (numeric and unrelated overrides do not count)
    annotated = get_inner_parameters_v2(petab_problem)
    coupled_pars = set()
    for measurement in petab_problem.measurements:
        group = tuple(
            override
            for override in map(str, measurement.observable_parameters)
            if override in annotated
        )
        if len(group) >= 2 and {
            InnerParameterType.SCALING,
            InnerParameterType.OFFSET,
        } <= {annotated[override] for override in group}:
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
                if coupled_parameter_id not in id_to_par:
                    # only a fully estimated pair can be solved analytically
                    raise NotImplementedError(
                        "Coupled scaling/offset parameters must all be "
                        f"estimated, but `{coupled_parameter_id}` (coupled to "
                        f"`{par.inner_parameter_id}`) is not estimated."
                    )
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
    from ..petab import get_inner_parameters_v2

    inner_parameter_types = get_inner_parameters_v2(petab_problem)

    parameters = []
    for parameter in petab_problem.parameters:
        if not parameter.estimate or parameter.id not in inner_parameter_types:
            continue
        inner_parameter_type = inner_parameter_types[parameter.id]

        # all inner parameter types but scaling and offset require fixed
        #  bounds (cf. `correct_parameter_df_bounds`)
        lb, ub = parameter.lb, parameter.ub
        if inner_parameter_type not in (
            InnerParameterType.SCALING,
            InnerParameterType.OFFSET,
        ):
            bounds = INNER_PARAMETER_BOUNDS[inner_parameter_type]
            lb, ub = bounds[PYPESTO_LOWER_BOUND], bounds[PYPESTO_UPPER_BOUND]

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

    return parameters


def ixs_for_measurement_specific_parameters_v2(
    petab_problem: "v2.Problem",
    amici_model: "asd.Model",
    x_ids: list[str],
) -> dict[str, list[tuple[int, int, int]]]:
    """
    Create mapping of parameters to measurements for a PEtab v2 problem.

    See :func:`ixs_for_measurement_specific_parameters`. The condition index
    is the position of the experiment in ``petab_problem.experiments``, which
    is the order of the ``ExpData`` objects created by
    :class:`amici.sim.sundials.petab.ExperimentManager`.
    """
    ixs_for_par = {}
    observable_ids = amici_model.get_observable_ids()
    measurement_df = petab_problem.measurement_df

    for condition_ix, experiment in enumerate(petab_problem.experiments):
        _ixs_for_condition(
            measurement_df[measurement_df[EXPERIMENT_ID] == experiment.id],
            condition_ix,
            observable_ids,
            x_ids,
            ixs_for_par,
        )
    return ixs_for_par
