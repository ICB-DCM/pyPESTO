import numpy as np
from typing import List
import pandas as pd

from ...C import (
    PARAMETER_TYPE,
    PARAMETER_GROUP,
    PARAMETER_CATEGORY,
    InnerParameterType,
)
from ..problem import InnerProblem
from ..problem import (
    inner_parameters_from_parameter_df,
    ixs_for_measurement_specific_parameters,
    ix_matrices_from_arrays,
)
from .optimal_scaling_parameter import OptimalScalingParameter

try:
    import amici
    import petab
    from petab.C import (
        ESTIMATE,
        LOWER_BOUND,
        PARAMETER_ID,
        PARAMETER_SCALE,
        UPPER_BOUND,
    )
except ImportError:
    pass


class OptimalScalingProblem(InnerProblem):
    def __init__(
        self,
        xs: List[OptimalScalingParameter],
        # hard_constraints: pd.DataFrame,
        data: List[np.ndarray],
    ):
        super().__init__(xs, data)
        # self.hard_constraints = hard_constraints
        self.groups = {}

        for idx, gr in enumerate(
            self.get_groups_for_xs(InnerParameterType.OPTIMALSCALING)
        ):
            self.groups[gr] = {}
            xs = self.get_xs_for_group(gr)
            self.groups[gr]['num_categories'] = len(xs)
            self.groups[gr]['num_datapoints'] = np.sum(
                [np.sum([np.sum(ixs) for ixs in x.ixs]) for x in xs]
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

            self.groups[gr]['cat_ixs'] = {}
            self.get_cat_indices(gr, xs)

            self.groups[gr]['C'] = self.initialize_c(gr, xs)

            self.groups[gr]['W'] = self.initialize_w(gr)

            self.groups[gr]['Wdot'] = self.initialize_w(gr)

    @staticmethod
    def from_petab_amici(
        petab_problem: petab.Problem,
        amici_model: 'amici.Model',
        edatas: List['amici.ExpData'],
    ):
        return qualitative_inner_problem_from_petab_problem(
            petab_problem, amici_model, edatas
        )

    def get_groups_for_xs(self, inner_parameter_type: str) -> List[int]:
        """Get unique list of ``OptimalScalingParameter.group`` values."""
        try:
            groups = [
                x.group for x in self.get_xs_for_type(inner_parameter_type)
            ]
        except:
            breakpoint()
        return list(set(groups))

    # FIXME does this break if there's inner parameters (xs) of different sorts, i.e.
    # not only optimalscaling? Think so...
    def get_xs_for_group(self, group: int) -> List[OptimalScalingParameter]:
        """Get ``OptimalScalingParameter``s that belong to the given group."""
        return [x for x in self.xs.values() if x.group == group]

    def initialize_c(self, gr, xs):
        constr = np.zeros(
            [
                self.groups[gr]['num_constr_full'],
                self.groups[gr]['num_inner_params'],
            ]
        )
        data_idx = 0
        for cat_idx, x in enumerate(xs):
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

    def get_cat_for_xi_idx(self, gr, data_idx):
        for cat_idx, (_, indices) in enumerate(
            self.groups[gr]['cat_ixs'].items()
        ):
            if data_idx in indices:
                return cat_idx

    def get_cat_indices(self, gr, xs):
        idx_tot = 0
        for x in xs:
            num_points = np.sum(
                [np.sum(x.ixs[idx]) for idx in range(len(x.ixs))]
            )
            self.groups[gr]['cat_ixs'][x.inner_parameter_id] = list(
                range(idx_tot, idx_tot + num_points)
            )
            idx_tot += num_points

    def get_last_category_for_group(self, gr):
        last_category = 1
        for x in self.xs.values():
            if x.group == gr and x.category > last_category:
                last_category = x.category
        return last_category

    # def get_hard_constraints_for_group(self, group: float):
    #     return self.hard_constraints[self.hard_constraints['group'].astype(float)==group]


def qualitative_inner_problem_from_petab_problem(
    petab_problem: petab.Problem,
    amici_model: 'amici.Model',
    edatas: List['amici.ExpData'],
):
    # get hard constrained measurements from measurement.df
    # hard_constraints=get_hard_constraints(petab_problem)

    # inner parameters
    inner_parameters = qualitative_inner_parameters_from_parameter_df(
        petab_problem.parameter_df
    )

    x_ids = [x.inner_parameter_id for x in inner_parameters]

    # used indices for all measurement specific parameters
    ixs = ixs_for_measurement_specific_parameters(
        petab_problem, amici_model, x_ids
    )
    # print("ixs : \n", ixs)
    # transform experimental data
    edatas = [
        amici.numpy.ExpDataView(edata)['observedData'] for edata in edatas
    ]
    # print("edatas : \n",edatas)
    # matrixify
    ix_matrices = ix_matrices_from_arrays(ixs, edatas)
    # print("ix_matrices : \n",ix_matrices)
    # assign matrices
    for par in inner_parameters:
        par.ixs = ix_matrices[par.inner_parameter_id]

    return OptimalScalingProblem(
        inner_parameters,
        # hard_constraints,
        edatas,
    )


def qualitative_inner_parameters_from_parameter_df(
    df: pd.DataFrame,
) -> List[OptimalScalingParameter]:
    """
    Create list of inner free parameters from PEtab parameter table.

    Inner parameters are those that have a non-empty `parameterType` in the
    PEtab problem.
    """
    # create list of hierarchical parameters
    df = df.reset_index()

    for col in (PARAMETER_TYPE, PARAMETER_GROUP, PARAMETER_CATEGORY):
        if col not in df:
            df[col] = None

    parameters = []

    for _, row in df.iterrows():
        if not row[ESTIMATE]:
            continue
        if petab.is_empty(row[PARAMETER_TYPE]):
            continue
        parameters.append(
            OptimalScalingParameter(
                inner_parameter_id=row[PARAMETER_ID],
                inner_parameter_type=row[PARAMETER_TYPE],
                scale=row[PARAMETER_SCALE],
                lb=row[LOWER_BOUND],
                ub=row[UPPER_BOUND],
                category=row[PARAMETER_CATEGORY],
                group=row[PARAMETER_GROUP],
            )
        )

    return parameters


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
