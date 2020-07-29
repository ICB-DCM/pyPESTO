import petab
import amici
from typing import Union, Optional, Iterable, List
from ..problem import Problem
from ..result import Result


# should we specify Optimizer here?
class DesignProblem(dict):
    """
    The problem formulation for an experimental design setting

    Parameters
    ----------
    problem_list:
        list of dicts, each dict with entries 'condition_df', 'observable_df',
        'measurement_df' which specify rows to add to the existing dataframes
    model:
        the amici model
    problem:
        the pypesto problem for the initial setting
    result:
        the pypesto result for the initial setting
    petab_problem:
        the petab problem for the initial setting
    x:
        the parameter to be used for the forward simulation to generate new
        data
    n_optimize_runs:
        the number of multistart optimization runs to be done after a new
        condition has been added
    n_cond_to_add:
        how many new conditions should be added in the end
    criteria_list:
        list of criteria names, specifies which criteria values should be
        computed
    const_for_hess:
        a constant that can be added to the eigenvalues before computing the
        criteria values
    profiles:
        if a criteria based on profiles eg length of confidence interval
        should be computed
    number_of_measurements:
        how many new measurements (with different noise added) are to be
        added to the measurement dataframe after finding the exact value via
        the forward simulation
    """

    def __init__(self,
                 problem_list: list,
                 model: Union['amici.Model', 'amici.ModelPtr'],
                 problem: Problem,
                 result: Result,
                 petab_problem: petab.Problem,
                 x: Optional[Iterable[float]] = None,
                 n_optimize_runs: int = 10,
                 n_cond_to_add: int = 1,
                 criteria_list: List[str] = None,
                 chosen_criteria: str = 'det',
                 const_for_hess: float = 10 ** (-4),
                 profiles: bool = False,
                 number_of_measurements: int = 3):

        super().__init__()

        if criteria_list is None:
            criteria_list = ['det', 'trace', 'ratio', 'rank', 'eigmin',
                             'number_good_eigvals']
        self.problem_list = problem_list
        self.model = model
        self.problem = problem
        self.result = result
        self.petab_problem = petab_problem
        self.x = x
        self.n_optimize_runs = n_optimize_runs
        self.n_cond_to_add = n_cond_to_add
        self.criteria_list = criteria_list
        self.chosen_criteria = chosen_criteria
        self.const_for_hess = const_for_hess
        self.profiles = profiles
        self.number_of_measurements = number_of_measurements

        # TODO x, profiles, number_of_measurements are not actively used in
        #  the code right now

        # sanity checks for lengths of df in problem_list
        if not self.problem_list:
            raise ValueError('you need to pass a nonempty list of candidates')
        for dict in self.problem_list:
            if len(dict['condition_df'].columns) != len(
                    self.petab_problem.condition_df.columns):
                raise ValueError(
                    'condition dataframe in given candidates has wrong length')
            if dict['observable_df'] is not None and len(
                    dict['observable_df'].columns) != len(
                    self.petab_problem.observable_df.columns):
                raise ValueError(
                    'observable dataframe in given candidates has wrong '
                    'length')
            if len(dict['measurement_df'].columns) != len(
                    self.petab_problem.measurement_df.columns):
                raise ValueError(
                    'measurement dataframe in given candidates has wrong '
                    'length')

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
