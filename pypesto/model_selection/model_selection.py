from typing import Dict, List

import petab
import numpy as np
from ..problem import Problem

MODEL_ID = "ModelId"


class ModelSelectionProblem:
    """

    """
    def __init__(self, petab_problem: petab.problem, row: Dict[str, float]):
        self.pypesto_problem = None  # self.row2problem(petab_problem, row)
        self.optimization_result = None
        self.id = row[MODEL_ID]
        self.AIC = None
        self.BIC = None
        self.configuration = row

    def get_aic(self):
        raise NotImplementedError()

    def get_bic(self):
        raise NotImplementedError()

    def row2problem(self, petab_problem, row) -> Problem:
        # in progress
        raise NotImplementedError()

    def calibrate(self):
        # TODO
        # self.get_aic()
        # self.get_bic()
        self.AIC = self.configuration['AIC']


class ModelSelection:
    """
    here it is assumed that that there is only one petab_problem
    """
    def __init__(self, petab_problem: petab.problem,
                 model_selection_config: str):
        self.petab_problem = petab_problem
        # unpacked_ms_conf = unpack_file(model_selection_config) # todo iterator
        self.unpacked_ms_conf = [
            {'ModelId': 'M1_0', 'Parameter1': 1, 'Parameter2': 1, 'AIC': 3},
            {'ModelId': 'M2_0', 'Parameter1': 1, 'Parameter2': 0, 'AIC': 3},
            {'ModelId': 'M3_0', 'Parameter1': 0, 'Parameter2': 0, 'AIC': 5},
            {'ModelId': 'M3_1', 'Parameter1': 0, 'Parameter2': 1, 'AIC': 2},
            {'ModelId': 'M3_2', 'Parameter1': 1, 'Parameter2': 0, 'AIC': 3},
            {'ModelId': 'M3_3', 'Parameter1': 0, 'Parameter2': 0, 'AIC': 1},
            {'ModelId': 'M3_4', 'Parameter1': float("Nan"), 'Parameter2': 0,
             'AIC': 13},
            {'ModelId': 'M3_5', 'Parameter1': float("Nan"),
             'Parameter2': float("Nan"), 'AIC': 1}]
        self.final_model = None
        self.parameter_names = ['Parameter1', 'Parameter2']
        self.selection_history = []

    def get_smallest_order_problem(self) -> ModelSelectionProblem:

        smallest_order_problem = None
        for model_descr in self.unpacked_ms_conf:
            if not smallest_order_problem:
                smallest_order_problem = model_descr
            else:
                for par in self.parameter_names:
                    if (not np.isnan(model_descr[par]) and
                        np.isnan(smallest_order_problem[par])) \
                            or smallest_order_problem[par] > model_descr[par]:
                        smallest_order_problem = model_descr
                        break
        return ModelSelectionProblem(self.petab_problem,
                                     smallest_order_problem)

    def get_largest_problem(self) -> ModelSelectionProblem:
        raise NotImplementedError()

    def forward_selection(self, criterion='AIC'):
        ms_problem = self.get_smallest_order_problem()  # check also history?
        self.selection_history.append(ms_problem.id)
        ms_problem.calibrate()
        proceed = True

        while proceed:
            proceed = False
            model_candidate_indices = self.get_next_step_candidates(
                ms_problem.configuration)
            better_model_id = None
            for ind in model_candidate_indices:
                candidate_model = ModelSelectionProblem(
                        self.petab_problem, self.unpacked_ms_conf[ind])
                candidate_model.calibrate()
                if criterion == 'AIC':
                    if candidate_model.AIC < ms_problem.AIC:
                        ms_problem = candidate_model
                        better_model_id = candidate_model.id
                        proceed = True
            if better_model_id:
                self.selection_history.append(better_model_id)
        return ms_problem

    def backward_selection(self, criterion='AIC'):
        raise NotImplementedError()

    def get_next_step_candidates(self, conf_dict: Dict[str, float],
                                 direction=0) -> List[int]:
        """
        returns indices of models that should be considered on the next step
        for now just direction = 0 is considered

        Parameters
        ----------
        conf_dict:
        direction: 0 - forward, 1 - backward

        Returns
        -------

        """
        rel_complexity_orders = []
        for model_descr in self.unpacked_ms_conf:
            rel_complexity_orders.append(0)
            for par in self.parameter_names:
                if (not np.isnan(model_descr[par]) and
                    np.isnan(conf_dict[par])) \
                        or conf_dict[par] > model_descr[par]:
                    rel_complexity_orders[-1] -= 1
                if (np.isnan(model_descr[par]) and
                    not np.isnan(conf_dict[par])) \
                        or conf_dict[par] < model_descr[par]:
                    rel_complexity_orders[-1] += 1
        next_step_order = min(i for i in rel_complexity_orders if i > 0)
        return [i for i, value in enumerate(rel_complexity_orders) if
                value == next_step_order]
