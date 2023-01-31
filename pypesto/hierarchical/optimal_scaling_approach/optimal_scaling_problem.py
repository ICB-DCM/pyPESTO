from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ...C import (
    INNER_PARAMETER_BOUNDS,
    LIN,
    MEASUREMENT_CATEGORY,
    MEASUREMENT_GROUP,
    MEASUREMENT_TYPE,
    ORDINAL,
    STANDARD,
    TIME,
    InnerParameterType,
)
from ..problem import (
    InnerProblem,
    _get_timepoints_with_replicates,
    ix_matrices_from_arrays,
)
from .optimal_scaling_parameter import OptimalScalingParameter

try:
    import amici
    import petab
    from petab.C import OBSERVABLE_ID
except ImportError:
    pass


class OptimalScalingProblem(InnerProblem):
    """Class of the Optimal Scaling inner subproblem."""

    def __init__(
        self,
        xs: List[OptimalScalingParameter],
        # hard_constraints: pd.DataFrame,
        data: List[np.ndarray],
        method: str,
    ):
        """Construction of the Optimal Scaling inner subproblem.

        Parameters
        ----------
            xs:
                List of `OptimalScalingParameter`s of the subproblem.
            data:
                The data of the problem.
            method:
                A string representing the method of the Optimal Scaling approach, either 'reduced' or 'standard'.
        """
        super().__init__(xs, data)
        # self.hard_constraints = hard_constraints
        self.groups = {}
        self.method = method

        for idx, gr in enumerate(
            self.get_groups_for_xs(InnerParameterType.OPTIMALSCALING)
        ):
            self.groups[gr] = {}
            xs = self.get_xs_for_group(gr)
            self.groups[gr]['num_categories'] = len(
                set([x.category for x in xs])
            )
            self.groups[gr]['num_datapoints'] = np.sum(
                [
                    np.sum([np.sum(ixs) for ixs in x.ixs])
                    for x in self.get_cat_ub_parameters_for_group(gr)
                ]
            )

            self.groups[gr]['surrogate_data'] = np.zeros(
                self.groups[gr]['num_datapoints']
            )

            self.groups[gr]['num_inner_params'] = (
                self.groups[gr]['num_datapoints']
                + 2 * self.groups[gr]['num_categories']
            )

            self.groups[gr]['num_constr_full'] = (
                2 * self.groups[gr]['num_datapoints']
                + 2 * self.groups[gr]['num_categories']
            )  # - 1

            self.groups[gr]['lb_indices'] = list(
                range(
                    self.groups[gr]['num_datapoints'],
                    self.groups[gr]['num_datapoints']
                    + self.groups[gr]['num_categories'],
                )
            )

            self.groups[gr]['ub_indices'] = list(
                range(
                    self.groups[gr]['num_datapoints']
                    + self.groups[gr]['num_categories'],
                    self.groups[gr]['num_inner_params'],
                )
            )

            self.groups[gr]['C'] = self.initialize_c(gr)

            self.groups[gr]['W'] = self.initialize_w(gr)

            self.groups[gr]['Wdot'] = self.initialize_w(gr)

    @staticmethod
    def from_petab_amici(
        petab_problem: petab.Problem,
        amici_model: 'amici.Model',
        edatas: List['amici.ExpData'],
        method: str,
    ):
        """Construct the inner problem from the `petab_problem`."""
        return qualitative_inner_problem_from_petab_problem(
            petab_problem, amici_model, edatas, method
        )

    def get_groups_for_xs(self, inner_parameter_type: str) -> List[int]:
        """Get unique list of ``OptimalScalingParameter.group`` values."""
        groups = [x.group for x in self.get_xs_for_type(inner_parameter_type)]
        return list(set(groups))

    # FIXME does this break if there's inner parameters (xs) of different sorts, i.e.
    # not only optimalscaling? Think so...
    def get_xs_for_group(self, group: int) -> List[OptimalScalingParameter]:
        """Get ``OptimalScalingParameter``s that belong to the given group."""
        return [x for x in self.xs.values() if x.group == group]

    def get_free_xs_for_group(
        self, group: int
    ) -> List[OptimalScalingParameter]:
        """Get ``OptimalScalingParameter``s that are free and belong to the
        given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.estimate is True
        ]

    def get_fixed_xs_for_group(
        self, group: int
    ) -> List[OptimalScalingParameter]:
        """Get ``OptimalScalingParameter``s that are fixed and belong to the
        given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.estimate is False
        ]

    def get_cat_ub_parameters_for_group(
        self, group: int
    ) -> List[OptimalScalingParameter]:
        """Get ``OptimalScalingParameter``s that are category upper boundaries
        and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.inner_parameter_id[:6] == 'cat_ub'
        ]

    def get_cat_lb_parameters_for_group(
        self, group: int
    ) -> List[OptimalScalingParameter]:
        """Get ``OptimalScalingParameter``s that are category lower boundaries
        and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.inner_parameter_id[:6] == 'cat_lb'
        ]

    def initialize_c(self, gr):
        """Initialize the constraints matrix C for the group 'gr'."""
        constr = np.zeros(
            [
                self.groups[gr]['num_constr_full'],
                self.groups[gr]['num_inner_params'],
            ]
        )
        data_idx = 0
        for cat_idx, x in enumerate(self.get_cat_ub_parameters_for_group(gr)):
            num_data_in_cat = int(
                np.sum([np.sum(x.ixs[idx]) for idx in range(len(x.ixs))])
            )
            for data_in_cat_idx in range(num_data_in_cat):
                # x_lower - y_surr <= 0
                constr[data_idx, data_idx] = -1
                constr[
                    data_idx, cat_idx + self.groups[gr]['num_datapoints']
                ] = 1

                # y_surr - x_upper <= 0
                constr[
                    data_idx + self.groups[gr]['num_datapoints'], data_idx
                ] = 1
                constr[
                    data_idx + self.groups[gr]['num_datapoints'],
                    cat_idx
                    + self.groups[gr]['num_datapoints']
                    + self.groups[gr]['num_categories'],
                ] = -1
                data_idx += 1

                # x_upper_i - x_lower_{i+1} <= 0
            if cat_idx == 0:  # - 1:
                constr[
                    2 * self.groups[gr]['num_datapoints'] + cat_idx,
                    self.groups[gr]['num_datapoints'] + cat_idx,
                ] = -1
            else:
                constr[
                    2 * self.groups[gr]['num_datapoints'] + cat_idx,
                    self.groups[gr]['num_datapoints'] + cat_idx,
                ] = -1  # + 1] = -1
                constr[
                    2 * self.groups[gr]['num_datapoints'] + cat_idx,
                    self.groups[gr]['num_datapoints']
                    + self.groups[gr]['num_categories']
                    + cat_idx
                    - 1,
                ] = 1  # + cat_idx] = 1

            constr[
                2 * self.groups[gr]['num_datapoints']
                + self.groups[gr]['num_categories']  # - 1
                + cat_idx,
                self.groups[gr]['lb_indices'][cat_idx],
            ] = 1
            constr[
                2 * self.groups[gr]['num_datapoints']
                + self.groups[gr]['num_categories']  # - 1
                + cat_idx,
                self.groups[gr]['ub_indices'][cat_idx],
            ] = -1

        return constr

    def initialize_w(self, gr):
        """Initialize the weight matrix W for the group 'gr'."""
        weights = np.diag(
            np.block(
                [
                    np.ones(self.groups[gr]['num_datapoints']),
                    np.zeros(2 * self.groups[gr]['num_categories']),
                ]
            )
        )
        return weights

    def get_w(self, gr, y_sim_all):
        """Returns the weight matrix W of the group 'gr'."""
        weights = np.diag(
            np.block(
                [
                    np.ones(self.groups[gr]['num_datapoints'])
                    / (np.sum(np.abs(y_sim_all)) + 1e-8),
                    np.zeros(2 * self.groups[gr]['num_categories']),
                ]
            )
        )
        return weights

    def get_wdot(self, gr, y_sim_all, sy_all):
        """Returns the derivative of the weight matrix W of the group 'gr' with
        respect to a outer parameter."""
        weights = np.diag(
            np.block(
                [
                    np.ones(self.groups[gr]['num_datapoints'])
                    * (
                        -1
                        * np.sum(sy_all)
                        / ((np.sum(np.abs(y_sim_all)) + 1e-8) ** 2)
                    ),
                    np.zeros(2 * self.groups[gr]['num_categories']),
                ]
            )
        )
        return weights

    def get_d(self, gr, xs, y_sim_all, eps):
        """Returns vector d of minimal gaps and ranges."""
        # if 'minGap' not in options:
        #    eps = 1e-16
        # else:
        #    eps = options['minGap']
        max_simulation = np.nanmax(y_sim_all)

        interval_range = max_simulation / (2 * len(xs) + 1)
        interval_gap = max_simulation / (4 * (len(xs) - 1) + 1)
        # if interval_gap < eps:
        #   interval_gap = eps

        d = np.zeros(self.groups[gr]['num_constr_full'])

        d[
            2 * self.groups[gr]['num_datapoints']
            + 1 : 2 * self.groups[gr]['num_datapoints']
            + self.groups[gr]['num_categories']
        ] = (
            interval_gap + eps
        )  # - 1] \

        d[
            2 * self.groups[gr]['num_datapoints']
            + self.groups[gr]['num_categories'] :
        ] = interval_range
        return d

    def get_dd_dtheta(self, gr, xs, y_sim_all, sy_all):
        """Returns the derivative of vector d of minimal gaps and ranges with
        respect to a outer parameter."""
        max_sim_idx = np.argmax(y_sim_all)
        max_sy = sy_all[max_sim_idx]
        dd_dtheta = np.zeros(self.groups[gr]['num_constr_full'])

        dinterval_range_dtheta = max_sy / (2 * len(xs) + 1)
        dinterval_gap_dtheta = max_sy / (4 * (len(xs) - 1) + 1)

        dd_dtheta[
            2 * self.groups[gr]['num_datapoints']
            + 1 : 2 * self.groups[gr]['num_datapoints']
            + self.groups[gr]['num_categories']
        ] = dinterval_gap_dtheta

        dd_dtheta[
            2 * self.groups[gr]['num_datapoints']
            + self.groups[gr]['num_categories'] :
        ] = dinterval_range_dtheta

        return dd_dtheta

    def get_inner_parameter_dictionary(self):
        """Returns a dictionary with inner parameter ids and their values."""
        inner_par_dict = {}
        for x_id, x in self.xs.items():
            inner_par_dict[x_id] = x.value
        return inner_par_dict

    # def get_last_category_for_group(self, gr):
    #     last_category = 1
    #     for x in self.xs.values():
    #         if x.group == gr and x.category > last_category:
    #             last_category = x.category
    #     return last_category

    # def get_hard_constraints_for_group(self, group: float):
    # return
    # self.hard_constraints[self.hard_constraints['group'].astype(float)==group]


def qualitative_inner_problem_from_petab_problem(
    petab_problem: petab.Problem,
    amici_model: 'amici.Model',
    edatas: List['amici.ExpData'],
    method: str,
):
    """Constructs the inner problem from the `petab_problem`."""
    # get hard constrained measurements from measurement.df
    # hard_constraints=get_hard_constraints(petab_problem)

    # inner parameters
    inner_parameters = qualitative_inner_parameters_from_measurement_df(
        petab_problem.measurement_df, method
    )

    # used indices for all measurement specific parameters
    ixs = qualitatiave_ixs_for_measurement_specific_parameters(
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

    return OptimalScalingProblem(
        inner_parameters,
        # hard_constraints,
        edatas,
        method,
    )


def qualitative_inner_parameters_from_measurement_df(
    df: pd.DataFrame,
    method: str,
) -> List[OptimalScalingParameter]:
    """Create list of inner free parameters from PEtab measurement table
    dependent on the method provided."""
    # create list of hierarchical parameters
    df = df.reset_index()

    # FIXME Make validate PEtab for optimal scaling
    # for col in (MEASUREMENT_TYPE, MEASUREMENT_GROUP, MEASUREMENT_CATEGORY):
    #     if col not in df:
    #         df[col] = None
    # if petab.is_empty(row[PARAMETER_TYPE]):
    #     continue

    estimate = get_estimate_for_method(method)
    par_types = ['cat_lb', 'cat_ub']

    inner_parameters = []
    lb, ub = INNER_PARAMETER_BOUNDS[InnerParameterType.OPTIMALSCALING].values()

    for par_type, par_estimate in zip(par_types, estimate):
        for _, row in df.iterrows():
            if row[MEASUREMENT_TYPE] == ORDINAL:
                par_id = f'{par_type}_{row[OBSERVABLE_ID]}_{row[MEASUREMENT_GROUP]}_{row[MEASUREMENT_CATEGORY]}'

                # Create only one set of bound parameters per category of a
                # group.
                if par_id not in [
                    inner_par.inner_parameter_id
                    for inner_par in inner_parameters
                ]:
                    inner_parameters.append(
                        OptimalScalingParameter(
                            inner_parameter_id=par_id,
                            inner_parameter_type=InnerParameterType.OPTIMALSCALING,
                            scale=LIN,
                            lb=lb,
                            ub=ub,
                            category=row[MEASUREMENT_CATEGORY],
                            group=row[MEASUREMENT_GROUP],
                            estimate=par_estimate,
                        )
                    )
    inner_parameters.sort(key=lambda x: (x.group, x.category))

    return inner_parameters


def get_estimate_for_method(method: str):
    """Returns which inner parameters to estimate dependent on the method
    provided."""
    estimate_ub = True
    estimate_lb = False

    if method == STANDARD:
        estimate_lb = True

    return estimate_lb, estimate_ub


def qualitatiave_ixs_for_measurement_specific_parameters(
    petab_problem: 'petab.Problem',
    amici_model: 'amici.Model',
    inner_parameters: List[OptimalScalingParameter],
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Create mapping of parameters to measurements.

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

                inner_par_ids_for_meas = get_inner_par_ids_for_measurement(
                    measurement, inner_parameters
                )

                # try to insert if hierarchical parameter
                for override in inner_par_ids_for_meas:
                    ixs_for_par.setdefault(override, []).append(
                        (condition_ix, time_w_reps_ix, observable_ix)
                    )
    return ixs_for_par


def get_inner_par_ids_for_measurement(
    measurement: Dict,
    inner_parameters: List[OptimalScalingParameter],
):
    """Returns inner parameter ids of parameters which are related to the
    measurement."""
    return [
        inner_par.inner_parameter_id
        for inner_par in inner_parameters
        if inner_par.category == measurement[MEASUREMENT_CATEGORY]
        and inner_par.group == measurement[MEASUREMENT_GROUP]
    ]


# def get_hard_constraints(petab_problem: petab.Problem):
#     measurement_df = petab_problem.measurement_df
#     parameter_df = petab_problem.parameter_df
#     hard_cons_df=pd.DataFrame(columns=['observableId', 'measurement', 'group', 'category', 'seen']) #ADD CONDITION HERE?
#     if('IsHardConstraint' in measurement_df):
#         for i in range(len(measurement_df)):
#             if(measurement_df.loc[i, "IsHardConstraint"]==True):
#                 group=float(parameter_df[parameter_df['parameterName']==measurement_df.loc[i, "observableParameters"]]['parameterGroup'])
#                 category=float(parameter_df[parameter_df['parameterName']==measurement_df.loc[i, "observableParameters"]]['parameterCategory'])
#                 #print(group)
#                 seen=str(group) + '&' + str(category)
#                 if(seen not in hard_cons_df['seen'].values):
#                     hard_cons_df= hard_cons_df.append({'observableId': measurement_df.loc[i, "observableId"],
#                                     'measurement': measurement_df.loc[i, "measurement"],
#                                     'group' : group,
#                                     'category': category,
#                                     'seen': str(group) + '&' + str(category)}, ignore_index=True)
#                 #print(hard_cons_df, sep='\n')
#     return hard_cons_df
