from typing import List, Union, Iterable
from .design_problem import DesignProblem
from heapq import nlargest, nsmallest
from .opt_design_helpers import get_design_result, combinations_gen, \
    get_average_result_dict
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
        self.single_runs_average = None
        self.combi_runs = combi_runs
        self.initial_result = initial_result
        self.best_value = []
        self.best_index = []
        self.list_of_combinations = None

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def get_criteria_values(self, criteria: str,
                            runs: str = 'single_runs',
                            x_index: Union[int, str] = None):
        result = None
        if x_index is None:
            if len(self.design_problem.initial_x) > 1:
                x_index = 'average'
            else:
                x_index = 0
        if runs == 'single_runs':
            if isinstance(x_index, int):
                result = [d[criteria] for d in self.single_runs[x_index]]
            elif x_index == 'average':
                result = [d[criteria] for d in self.single_runs_average]
            else:
                print("Could not get criteria values. Specify x_index as an "
                      "integer for the parameter values or 'average'")
        elif runs == 'combi_runs':
            result = [d[criteria] for d in self.combi_runs[criteria]]
        else:
            print("Could not get criteria values. Specify if it should be "
                  "from 'single_runs' or 'combi_runs'")

        return result

    def get_best_conditions(self,
                            index: Union[int, str] = None,
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
        index: specifies for which of the parameter sets the best criteria
               should be taken, either an integer or 'average' for the
               average result
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
        # defaults to the average result in the case of multiple parameter sets
        if index is None:
            if len(self.design_problem.initial_x) > 1:
                index = 'average'
            else:
                index = 0
        if runs == 'single_runs':
            if isinstance(index, int):
                values = [d[key] for d in self.single_runs[index]]
            elif index == 'average':
                values = [d[key] for d in self.single_runs_average]
            else:
                print("Could not get criteria values. Specify x_index as an "
                      "integer for the parameter values or 'average'")

        elif runs == 'combi_runs':
            # self.combi_runs for each criteria are unsorted
            # temporarily save a sorted version
            sorted_combi_runs = {}
            for criteria in self.combi_runs:
                sorted_combi_runs[criteria] = sorted(self.combi_runs[criteria],
                                                     key=lambda d: d[criteria],
                                                     reverse=True)

            values = [best[key] for best in sorted_combi_runs[key]]
            indices = [best['candidate'] for best in sorted_combi_runs[key]]
        else:
            raise ValueError("can't find the specified runs to get "
                             "conditions from")
        # some criteria values may be None if the simulation failed etc
        # write -inf, +inf respectively for the code to work
        # note '== None' and 'is None' behave differently here
        none_ind = np.where(np.array(values) == None)[0].tolist()

        if maximize:
            values = [float('-inf') if i in none_ind else value
                      for i, value in enumerate(values)]
            best_indices, best_values = map(list, zip(
                *nlargest(n_best, enumerate(values), key=lambda x: x[1])))
        else:
            values = [float('inf') if i in none_ind else value
                      for i, value in enumerate(values)]
            best_indices, best_values = map(list, zip(
                *nsmallest(n_best, enumerate(values), key=lambda x: x[1])))

        if runs == 'single_runs':
            best_combination = [self.single_runs[0][i]['candidate']['id'] for i
                                in best_indices]
        elif runs == 'combi_runs':
            best_combination = [indices[i] for i in best_indices]
        else:
            raise ValueError("can't find the specified runs to get "
                             "conditions from")
        return best_values, best_combination

    def get_combi_run_result(self,
                             list_of_combinations: list,
                             x: Iterable,
                             x_index: int,
                             criteria_min) \
            -> List[dict]:
        """
        computes the new criteria values etc in a dict after adding the new
        addition to the fim

        Parameters
        ----------
        list_of_combinations:
        criteria_min: dict which has all criteria as keys. for each criteria
        save the lowest value of the n_save_combi_result that we save
        x: current set of parameters used
        x_index: index of x in design_problem.initial_x

        Returns
        -------

        """

        combi_runs = []
        for combi in list_of_combinations:
            total_hess_additional = np.zeros(self.single_runs[0][0][
                                                 'hess_additional'].shape)
            # new_hess = self.initial_result['hess']
            for index in combi:
                total_hess_additional = np.hstack((total_hess_additional,
                                                   self.single_runs[x_index][
                                                       index][
                                                       'hess_additional']))

                # new_hess = new_hess + self.single_runs[x_index][index][
                #     'hess_additional']

            new_result = get_design_result(
                design_problem=self.design_problem,
                candidate=combi,
                x=x,
                hess=self.initial_result['hess'],
                hess_additional=total_hess_additional,
                initial_result=self.initial_result,
                combi_runs=self.combi_runs,
                skip=True,
                criteria_min=criteria_min, )
            combi_runs.append(new_result)

        return combi_runs

    def get_best_combi_index_pairs(self,
                                   criteria: str):
        """
        returns a list where each entry is a list of indices. Each list of
        indices describes a tested combination
        """
        list_of_combinations = [entry['candidate'] for entry in
                                self['combi_runs'][criteria]]
        return list_of_combinations

    def get_best_combi_index(self,
                             criteria: str):
        """
        similar to 'get_best_combi_index_pairs' but doesn't return a multiple
        lists of combinations but a list for all first entries, a list for
        all second entries etc.
        """
        list_of_combinations = self.get_best_combi_index_pairs(criteria)

        list_of_indices = [[combi[i] for combi in list_of_combinations] for
                           i in range(len(list_of_combinations[0]))]
        return list_of_indices

    def check_combinations(self,
                           list_of_combinations: Union[List[List[int]], int]):
        """
        main routine for checking and saving the criteria values for
        combinations of candidates which where previously computed the
        'run_exp_design'
        results are saved in self.combi_runs

        Parameters
        ----------
        list_of_combinations:
            can be a list where each entry is a list of indices describing
            which entries in self.design_problem.experiment_list should be
            paired
            can be an integer n, then the whole combinatorial spaces of n
            combinations will be checked

        Returns
        -------

        """
        # set up combi_runs as dictionary with empty list as entry for
        # each criteria
        self.combi_runs = {key: [] for key in
                           self.design_problem.non_modified_criteria}
        if self.design_problem.modified_criteria:
            for crit in self.design_problem.modified_criteria:
                self.combi_runs[crit + '_modified'] = []

        # gets combinations of indices for entries in 'experiment_list'
        if isinstance(list_of_combinations, int):
            generator = combinations_gen(elements=range(len(
                self.design_problem.experiment_list)),
                length=list_of_combinations)
        else:
            generator = (element for element in list_of_combinations)

        # keep track of the lowest value for each criteria in the list
        # initialize with -infinity
        crit_min = {}
        for crit in self.design_problem.non_modified_criteria:
            crit_min[crit] = float("inf")
        if self.design_problem.modified_criteria:
            for crit in self.design_problem.modified_criteria:
                crit_min[crit + '_modified'] = float("inf")

        # for each combination check if it is in the best n many
        # if yes override saved entry

        for combi in generator:
            dict_list = []
            for x_index, x in enumerate(self.design_problem.initial_x):
                dict_result = \
                    self.get_combi_run_result(list_of_combinations=[combi],
                                              x=x,
                                              x_index=x_index,
                                              criteria_min=crit_min)[0]
                dict_list.append(dict_result)

            average_design_result = get_average_result_dict(dict_list)

            if len(self.design_problem.initial_x) > 1:
                average_design_result['x'] = 'average'
                average_design_result['hess'] = 'average'
                average_design_result['eigvals'] = 'average'

            for crit in self.design_problem.non_modified_criteria:
                if len(self.combi_runs[crit]) < \
                        self.design_problem.n_save_combi_result:
                    self.combi_runs[crit].append(average_design_result)

                    # update lowest
                    if average_design_result[crit] < crit_min[crit]:
                        crit_min[crit] = average_design_result[crit]

                elif average_design_result[crit] > crit_min[crit]:
                    # (hopefully) fast way of getting the index
                    to_replace = dict(
                        (d[crit], dict(d, index=index)) for (index, d) in
                        enumerate(
                            self.combi_runs[crit])).get(
                        crit_min[crit])['index']
                    self.combi_runs[crit][
                        to_replace] = average_design_result

                    crit_min[crit] = np.array([saved[crit] for saved
                                               in self.combi_runs[crit]]).min()

            # same but for modified criteria
            for crit in self.design_problem.modified_criteria:
                crit_mod = crit + '_modified'
                if len(self.combi_runs[crit_mod]) < \
                        self.design_problem.n_save_combi_result:
                    self.combi_runs[crit_mod].append(average_design_result)

                    # update lowest
                    if average_design_result[crit_mod] < crit_min[crit_mod]:
                        crit_min[crit_mod] = average_design_result[
                            crit_mod]

                elif average_design_result[crit_mod] > crit_min[crit_mod]:
                    # (hopefully) fast way of getting the index
                    to_replace = dict(
                        (d[crit_mod], dict(d, index=index)) for (
                            index, d)
                        in enumerate(self.combi_runs[crit_mod])).get(
                        crit_min[crit_mod])['index']
                    self.combi_runs[crit_mod][to_replace] = \
                        average_design_result
                    crit_min[crit_mod] = np.array(
                        [saved[crit_mod]
                         for saved in
                         self.combi_runs[
                             crit_mod]]).min()
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
