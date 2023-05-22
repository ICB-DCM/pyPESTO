from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ...C import (
    CURRENT_SIMULATION,
    DATAPOINTS,
    EXPDATA_MASK,
    INNER_NOISE_PARS,
    INNER_PARAMETER_BOUNDS,
    LIN,
    MAX_DATAPOINT,
    MEASUREMENT_TYPE,
    MIN_DATAPOINT,
    N_SPLINE_PARS,
    NONLINEAR_MONOTONE,
    NUM_DATAPOINTS,
    OPTIMIZE_NOISE,
    PARAMETER_TYPE,
    SPLINE_PAR_TYPE,
    TIME,
    InnerParameterType,
)
from ..problem import (
    InnerProblem,
    _get_timepoints_with_replicates,
    ix_matrices_from_arrays,
    scale_value,
)
from .parameter import SplineInnerParameter

try:
    import amici
    import petab
    from petab.C import (
        ESTIMATE,
        LOWER_BOUND,
        NOISE_PARAMETERS,
        OBSERVABLE_ID,
        PARAMETER_ID,
        UPPER_BOUND,
    )
except ImportError:
    pass


class SplineInnerProblem(InnerProblem):
    """Inner optimization problem for spline approximation.

    Attributes
    ----------
    xs:
        Mapping of (inner) parameter ID to ``InnerParameters``.
    data:
        Measurement data. One matrix (`num_timepoints` x `num_observables`)
        per simulation condition. Missing observations as NaN.
    groups:
        A dictionary of the groups of the subproblem.
    spline_ratio:
        The ratio of the number of spline inner parameters and number of measurements for each group.
    """

    def __init__(
        self,
        xs: List[SplineInnerParameter],
        data: List[np.ndarray],
        spline_ratio: float = 0.5,
    ):
        """Construct."""
        super().__init__(xs, data)
        self.spline_ratio = spline_ratio

        if spline_ratio <= 0:
            raise ValueError("Spline ratio must be a positive float.")
        self._initialize_groups()

    def _initialize_groups(self) -> None:
        """Initialize the groups of the subproblem."""
        self.groups = {}
        for group in self.get_groups_for_xs(InnerParameterType.SPLINE):
            xs = self.get_xs_for_group(group)

            self.groups[group] = {}
            self.groups[group][N_SPLINE_PARS] = len({x.index for x in xs})
            self.groups[group][DATAPOINTS] = self.get_measurements_for_group(
                group
            )
            self.groups[group][NUM_DATAPOINTS] = len(
                self.groups[group][DATAPOINTS]
            )
            self.groups[group][MIN_DATAPOINT] = np.min(
                self.groups[group][DATAPOINTS]
            )
            self.groups[group][MAX_DATAPOINT] = np.max(
                self.groups[group][DATAPOINTS]
            )

            self.groups[group][EXPDATA_MASK] = xs[0].ixs
            self.groups[group][CURRENT_SIMULATION] = np.zeros(
                self.groups[group][NUM_DATAPOINTS]
            )
            self.groups[group][INNER_NOISE_PARS] = 1
            self.groups[group][OPTIMIZE_NOISE] = (
                len(self.get_noise_parameters_for_group(group)) > 0
            )

    def initialize(self) -> None:
        """Initialize the subproblem."""
        # Initialize all parameter values.
        for x in self.xs.values():
            x.initialize()

        # Initialize the groups.
        for group in self.get_groups_for_xs(InnerParameterType.SPLINE):
            self.groups[group][CURRENT_SIMULATION] = np.zeros(
                self.groups[group][NUM_DATAPOINTS]
            )
            self.groups[group][INNER_NOISE_PARS] = 1

    @staticmethod
    def from_petab_amici(
        petab_problem: petab.Problem,
        amici_model: 'amici.Model',
        edatas: List['amici.ExpData'],
        spline_ratio: float = None,
    ) -> 'SplineInnerProblem':
        """Construct the inner problem from the `petab_problem`."""
        if spline_ratio is None:
            spline_ratio = get_default_options()
        return spline_inner_problem_from_petab_problem(
            petab_problem, amici_model, edatas, spline_ratio
        )

    def get_groups_for_xs(self, inner_parameter_type: str) -> List[int]:
        """Get unique list of ``SplineParameter.group`` values."""
        groups = [x.group for x in self.get_xs_for_type(inner_parameter_type)]
        return list(set(groups))

    def get_xs_for_group(self, group: int) -> List[SplineInnerParameter]:
        r"""Get ``SplineParameter``\s that belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group
            and x.inner_parameter_type == InnerParameterType.SPLINE
        ]

    def get_free_xs_for_group(self, group: int) -> List[SplineInnerParameter]:
        r"""Get ``SplineParameter``\s that are free and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group
            and x.estimate is True
            and x.inner_parameter_type == InnerParameterType.SPLINE
        ]

    def get_fixed_xs_for_group(self, group: int) -> List[SplineInnerParameter]:
        r"""Get ``SplineParameter``\s that are fixed and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group
            and x.estimate is False
            and x.inner_parameter_type == InnerParameterType.SPLINE
        ]

    def get_noise_parameters_for_group(
        self, group: int
    ) -> SplineInnerParameter:
        r"""Get the ``SplineParameter``\ that is a noise parameters and belongs to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group
            and x.inner_parameter_type == InnerParameterType.SIGMA
        ]

    def get_inner_parameter_dictionary(self) -> Dict:
        """Get a dictionary with all inner parameter ids and their values."""
        inner_par_dict = {}
        for x_id, x in self.xs.items():
            inner_par_dict[x_id] = x.value
        return inner_par_dict

    def get_inner_noise_parameter_dictionary(self) -> Dict:
        """Get a dictionary with all noise inner parameter ids and their values."""
        inner_par_dict = {}
        for x in self.get_xs_for_type(InnerParameterType.SIGMA):
            inner_par_dict[x.inner_parameter_id] = x.value
        return inner_par_dict

    def get_measurements_for_group(self, gr) -> np.ndarray:
        """Get measurements for a group."""
        # Taking the ixs of first inner parameter since
        # all of them are the same for the same group.
        ixs = self.get_xs_for_group(gr)[0].ixs

        return np.concatenate(
            [
                self.data[condition_index][ixs[condition_index]]
                for condition_index in range(len(ixs))
            ]
        )

    def get_noise_dummy_values(self, scaled: bool) -> Dict[str, float]:
        """Get dummy values for noise parameters of the nonlinear-monotone observable."""
        return {
            x_id: scale_value(x.value, x.scale) if scaled else x.value
            for x_id, x in self.xs.items()
            if x.inner_parameter_type == InnerParameterType.SIGMA
        }


def get_default_options() -> Dict:
    """Return the default spline problem options dictionary."""
    spline_ratio = 1 / 2
    return spline_ratio


def spline_inner_problem_from_petab_problem(
    petab_problem: petab.Problem,
    amici_model: 'amici.Model',
    edatas: List['amici.ExpData'],
    spline_ratio: float = None,
):
    """Construct the inner problem from the `petab_problem`."""
    if spline_ratio is None:
        spline_ratio = get_default_options()
    elif spline_ratio <= 0:
        raise ValueError("Spline ratio must be a positive float.")

    # inner parameters
    inner_parameters = spline_inner_parameters_from_measurement_df(
        petab_problem.measurement_df, spline_ratio, amici_model
    )

    # noise parameters for nonlinear-monotone observables
    noise_parameters = noise_inner_parameters_from_parameter_df(
        petab_problem, amici_model
    )
    inner_parameters.extend(noise_parameters)

    # used indices for all measurement specific parameters
    ixs = spline_ixs_for_measurement_specific_parameters(
        petab_problem, amici_model, inner_parameters
    )

    # transform experimental data
    edatas = [
        amici.numpy.ExpDataView(edata)['observedData'] for edata in edatas
    ]

    # matrixify
    ix_matrices = ix_matrices_from_arrays(ixs, edatas)

    # assign matrices
    for par in inner_parameters:
        par.ixs = ix_matrices[par.inner_parameter_id]

    return SplineInnerProblem(
        inner_parameters,
        edatas,
        spline_ratio,
    )


def spline_inner_parameters_from_measurement_df(
    df: pd.DataFrame,
    spline_ratio: float,
    amici_model: 'amici.Model',
) -> List[SplineInnerParameter]:
    """Create list of inner free spline parameters from PEtab measurement table."""
    df = df.reset_index()

    observable_ids = amici_model.getObservableIds()

    par_type = SPLINE_PAR_TYPE
    estimate = True
    lb, ub = INNER_PARAMETER_BOUNDS[InnerParameterType.SPLINE].values()

    inner_parameters = []

    # Select the nonlinear monotone measurements.
    df = df[df[MEASUREMENT_TYPE] == NONLINEAR_MONOTONE]

    # Iterate over groups.
    for observable_id in observable_ids:
        group = observable_ids.index(observable_id) + 1
        df_for_group = df[df[OBSERVABLE_ID] == observable_id]

        n_spline_parameters = int(np.ceil(len(df_for_group) * spline_ratio))

        # Create n_spline_parameters number of spline inner parameters.
        for par_index in range(n_spline_parameters):
            par_id = f'{par_type}_{observable_id}_{group}_{par_index+1}'
            inner_parameters.append(
                SplineInnerParameter(
                    inner_parameter_id=par_id,
                    inner_parameter_type=InnerParameterType.SPLINE,
                    scale=LIN,
                    lb=lb,
                    ub=ub,
                    observable_id=observable_id,
                    group=group,
                    index=par_index + 1,
                    estimate=estimate,
                )
            )

    inner_parameters.sort(key=lambda x: (x.group, x.index))

    return inner_parameters


def noise_inner_parameters_from_parameter_df(
    petab_problem: 'petab.Problem',
    amici_model: 'amici.Model',
) -> List[SplineInnerParameter]:
    """Create list of inner free noise parameters from PEtab parameter table."""
    # Select the nonlinear monotone measurements.
    measurement_df = petab_problem.measurement_df
    measurement_df = measurement_df[
        measurement_df[MEASUREMENT_TYPE] == NONLINEAR_MONOTONE
    ]

    observable_ids = amici_model.getObservableIds()

    # Create a dictionary with unique pairs of observable id
    # and noise parameter from the measurement table.
    noise_parameter_to_observable = {}
    for _, row in measurement_df.iterrows():
        observable_id = row[OBSERVABLE_ID]
        noise_parameter_id = row[NOISE_PARAMETERS]
        noise_parameter_to_observable[noise_parameter_id] = observable_id

    noise_parameters = []

    parameter_df = petab_problem.parameter_df.reset_index()
    # Create noise inner parameters.
    for _, row in parameter_df.iterrows():
        if row[PARAMETER_ID] not in noise_parameter_to_observable:
            continue
        if petab.is_empty(row.get(PARAMETER_TYPE)) or not row[ESTIMATE]:
            continue
        observable_id = noise_parameter_to_observable[row[PARAMETER_ID]]
        group = observable_ids.index(observable_id) + 1

        noise_parameters.append(
            SplineInnerParameter(
                inner_parameter_id=row[PARAMETER_ID],
                inner_parameter_type=InnerParameterType.SIGMA,
                scale=LIN,
                lb=row[LOWER_BOUND],
                ub=row[UPPER_BOUND],
                observable_id=observable_id,
                group=group,
                index=None,
                estimate=True,
            )
        )

    return noise_parameters


def spline_ixs_for_measurement_specific_parameters(
    petab_problem: 'petab.Problem',
    amici_model: 'amici.Model',
    inner_parameters: List[SplineInnerParameter],
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Create mapping of parameters to measurements.

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

                inner_par_ids_for_meas = (
                    get_spline_inner_par_ids_for_measurement(
                        measurement, inner_parameters
                    )
                )

                # try to insert if hierarchical parameter
                for override in inner_par_ids_for_meas:
                    ixs_for_par.setdefault(override, []).append(
                        (condition_ix, time_w_reps_ix, observable_ix)
                    )
    return ixs_for_par


def get_spline_inner_par_ids_for_measurement(
    measurement: Dict,
    inner_parameters: List[SplineInnerParameter],
):
    """Return inner parameter ids of parameters which are related to the measurement."""
    return [
        inner_par.inner_parameter_id
        for inner_par in inner_parameters
        if inner_par.observable_id == measurement[OBSERVABLE_ID]
    ]
