from typing import List
from .design_problem import DesignProblem
from heapq import nlargest, nsmallest
from itertools import combinations
from .opt_design_helpers import get_design_result
import numpy as np


class DesignResult(dict):

    def __init__(self,
                 design_problem: DesignProblem,
                 single_runs: List[dict] = None,
                 combi_runs: List[dict] = None,
                 initial_result: List[dict] = None):

        super().__init__()

        if single_runs is None:
            single_runs = []
        self.design_problem = design_problem
        self.single_runs = single_runs
        self.combi_runs = combi_runs
        self.initial_result = initial_result
        self.best_value = None
        self.best_index = None
        self.list_of_combinations = None  # Union[List[List[int]], int]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def get_criteria_values(self, criteria: str, runs: str = None):
        result = None

        if runs is None or runs is 'single_runs':
            result = [d[criteria] for d in self.single_runs]
        elif runs is 'combi_runs':
            result = [d[criteria] for d in self.combi_runs]
        else:
            print("Could not get criteria values. Specify if it should be "
                  "from 'single_runs' or 'combi_runs'")

        return result

    def get_best_conditions(self,
                            key: str = None,
                            n_best: int = 1,
                            maximize: bool = True,
                            runs: str = 'single_runs') \
            -> (List[float], List[int]):
        """
        returns a tuple of the n_best many best values and indices from
        'experiment_list' for the specified criteria
        if n_cond_to_add > 1,  best_combination will be a list with the best
        indices to combine

        Parameters
        ----------
        key: the name of the criteria to compare
        n_best: the first n_best best will be chosen
        maximize: if the criteria is to be maximized or minimized
        runs: specifies if the results for the single candidates or
        combinations of these should be evaluated

        Returns
        -------
        best_values, best_combination:
            the tuple consisting of two lists with the best values for the
            criteria and the the list of the best indices
        """

        if key is None:
            key = self.design_problem.chosen_criteria

        if runs == 'single_runs':
            values = [d[key] for d in self.single_runs]
        elif runs == 'combi_runs':
            values = [d[key] for d in self.combi_runs]
        else:
            raise ValueError("can't find the specified runs to get "
                             "conditions from"
                             "")
        # some criteria values may be None if the simulation failed etc
        # write -inf, +inf respectively for the code to work
        none_ind = np.where(np.array(values) == None)[0].tolist()

        if maximize:
            values = [float('-inf') if i in none_ind else value for i, \
                      value in enumerate(values)]
            best_indices, best_values = map(list, zip(
                *nlargest(n_best, enumerate(values), key=lambda x: x[1])))
        else:
            values = [float('inf') if i in none_ind else value for i, \
                      value in enumerate(values)]
            best_indices, best_values = map(list, zip(
                *nsmallest(n_best, enumerate(values), key=lambda x: x[1])))

        if runs == 'single_runs':
            best_combination = [self.single_runs[i]['candidate']['id'] for i
                                in best_indices]
        elif runs == 'combi_runs':
            best_combination = [self.list_of_combinations[i] for i
                                in best_indices]
        else:
            raise ValueError("can't find the specified runs to get "
                             "conditions from")
        return best_values, best_combination

    def get_combi_run_result(self,
                             list_of_combinations: list):
        combi_result = []
        for combi in list_of_combinations:
            new_hess = self.initial_result['hess']
            for index in combi:
                new_hess = new_hess + self.single_runs[index][
                    'fim_addition']
            new_result = get_design_result(
                design_problem=self.design_problem,
                candidate=None,
                x=self.design_problem.initial_x,
                hess=new_hess)
            combi_result.append(new_result)

        return combi_result

    def check_combinations(self, list_of_combinations):

        if isinstance(list_of_combinations, int):
            index_combinations = list(combinations(range(len(
                self.design_problem.experiment_list)),
                list_of_combinations))
            index_combinations = [list(elem) for elem in
                                  index_combinations]
            self.list_of_combinations = index_combinations
        else:
            self.list_of_combinations = list_of_combinations

        self.combi_runs = self.get_combi_run_result(
            list_of_combinations=self.list_of_combinations)

        return self

    """
    # TODO implement as property ?
    @property
    def best_value(self):
        if self._best_value is None:
            self._best_value = self.get_best_conditions(
                self.design_problem.chosen_criteria)[0]
        return self._best_value

    @property
    def best_index(self):
        if self._best_index is None:
            self._best_index = self.get_best_conditions(
                self.design_problem.chosen_criteria)[1]
        return self._best_index
    """
