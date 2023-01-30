from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ...C import (
    INNER_PARAMETER_BOUNDS,
    LIN,
    MEASUREMENT_GROUP,
    MEASUREMENT_TYPE,
    NONLINEAR_MONOTONE,
    SPLINE_RATIO,
    TIME,
    InnerParameterType,
)
from ..problem import (
    InnerProblem,
    _get_timepoints_with_replicates,
    ix_matrices_from_arrays,
)
from .spline_parameter import SplineInnerParameter

try:
    import amici
    import petab
    from petab.C import OBSERVABLE_ID
except ImportError:
    pass


class SplineInnerProblem(InnerProblem):
    def __init__(
        self,
        xs: List[SplineInnerParameter],
        data: List[np.ndarray],
        spline_ratio: float = 1 / 2,
    ):
        super().__init__(xs, data)
        self.groups = {}
        self.spline_ratio = spline_ratio
        if spline_ratio <= 0:
            raise ValueError("Spline ratio must be a positive float.")

        for idx, gr in enumerate(
            self.get_groups_for_xs(InnerParameterType.SPLINE)
        ):
            xs = self.get_xs_for_group(gr)

            self.groups[gr] = {}
            self.groups[gr]['n_spline_pars'] = len(set([x.index for x in xs]))
            self.groups[gr]['datapoints'] = self.get_measurements_for_group(gr)
            self.groups[gr]['n_datapoints'] = len(
                self.groups[gr]['datapoints']
            )
            self.groups[gr]['min_datapoint'] = np.min(
                self.groups[gr]['datapoints']
            )
            self.groups[gr]['max_datapoint'] = np.max(
                self.groups[gr]['datapoints']
            )

            self.groups[gr]['expdata_mask'] = xs[0].ixs
            self.groups[gr]['current_simulation'] = np.zeros(
                self.groups[gr]['n_datapoints']
            )
            self.groups[gr]['noise_parameters'] = np.zeros(
                self.groups[gr]['n_datapoints']
            )

    @staticmethod
    def from_petab_amici(
        petab_problem: petab.Problem,
        amici_model: 'amici.Model',
        edatas: List['amici.ExpData'],
        options: Dict = None,
    ):
        if options is None:
            options = get_default_options()
        return qualitative_inner_problem_from_petab_problem(
            petab_problem, amici_model, edatas, options
        )

    def get_groups_for_xs(self, inner_parameter_type: str) -> List[int]:
        """Get unique list of ``SplineParameter.group`` values."""
        groups = [x.group for x in self.get_xs_for_type(inner_parameter_type)]
        return list(set(groups))

    # FIXME does this break if there's inner parameters (xs) of different sorts, i.e.
    # not only optimalscaling? Think so...
    def get_xs_for_group(self, group: int) -> List[SplineInnerParameter]:
        """Get ``SplineParameter``s that belong to the given group."""
        return [x for x in self.xs.values() if x.group == group]

    def get_free_xs_for_group(self, group: int) -> List[SplineInnerParameter]:
        """Get ``SplineParameter``s that are free and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.estimate is True
        ]

    def get_fixed_xs_for_group(self, group: int) -> List[SplineInnerParameter]:
        """Get ``SplineParameter``s that are fixed and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.estimate is False
        ]

    def get_inner_parameter_dictionary(self) -> Dict:
        """Get a dictionary with all inner parameter ids and their values."""
        inner_par_dict = {}
        for x_id, x in self.xs.items():
            inner_par_dict[x_id] = x.value
        return inner_par_dict

    def get_measurements_for_group(self, gr):
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


def get_default_options() -> Dict:
    return {SPLINE_RATIO: 1 / 2}


def qualitative_inner_problem_from_petab_problem(
    petab_problem: petab.Problem,
    amici_model: 'amici.Model',
    edatas: List['amici.ExpData'],
    options: Dict,
):
    spline_ratio = options[SPLINE_RATIO]

    # inner parameters
    inner_parameters = spline_inner_parameters_from_measurement_df(
        petab_problem.measurement_df, spline_ratio
    )

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
) -> List[SplineInnerParameter]:
    """
    Create list of inner free parameters from PEtab measurement table
    dependent on the spline_ratio provided (or default == 1/2).
    """
    # create list of hierarchical parameters
    df = df.reset_index()

    # FIXME Make validate PEtab for spline pars
    # for the same group all of the measurements have to be NONLINEAR_MONOTONE checke
    # check measurementType is in the df

    par_type = 'spline'
    estimate = True
    lb, ub = INNER_PARAMETER_BOUNDS[InnerParameterType.SPLINE].values()

    inner_parameters = []

    # Select the nonlinear monotone measurements.
    df = df[df[MEASUREMENT_TYPE] == NONLINEAR_MONOTONE]
    groups = list(set(df[MEASUREMENT_GROUP]))

    # Iterate over groups.
    for group in groups:
        df_for_group = df[df[MEASUREMENT_GROUP] == group]
        observable_id = df_for_group[OBSERVABLE_ID].values[0]
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
                    group=group,
                    index=par_index + 1,
                    estimate=estimate,
                )
            )

    inner_parameters.sort(key=lambda x: (x.group, x.index))

    return inner_parameters


def spline_ixs_for_measurement_specific_parameters(
    petab_problem: 'petab.Problem',
    amici_model: 'amici.Model',
    inner_parameters: List[SplineInnerParameter],
) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Create mapping of parameters to measurements.

    Returns
    -------
    A dictionary mapping parameter ID to a List of
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
    return [
        inner_par.inner_parameter_id
        for inner_par in inner_parameters
        if inner_par.group == measurement[MEASUREMENT_GROUP]
    ]
