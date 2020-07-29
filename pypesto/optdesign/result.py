from typing import List
from .design_problem import DesignProblem
from heapq import nlargest, nsmallest


class DesignResult(dict):

    def __init__(self,
                 design_problem: DesignProblem,
                 single_runs: List = None,
                 best_one=None):

        super().__init__()

        if single_runs is None:
            single_runs = []
        self.design_problem = design_problem
        self.single_runs = single_runs

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def get_criteria_values(self, criteria: str, run: int = 0):

        return [d[criteria] for d in self.single_runs[run]]

    def get_best_conditions(self,
                            key: str = None,
                            run: int = 0,
                            n_best: int = 1,
                            maximize: bool = True) \
            -> (List[float], List[int]):

        if key is None:
            key = self.design_problem.chosen_criteria

        values = [d[key] for d in self.single_runs[run]]

        if maximize:
            best_indices, best_values = map(list, zip(
                *nlargest(n_best, enumerate(values), key=lambda x: x[1])))
        else:
            best_indices, best_values = map(list, zip(
                *nsmallest(n_best, enumerate(values), key=lambda x: x[1])))

        return best_values, best_indices
