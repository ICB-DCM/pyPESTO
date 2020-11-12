from .design_problem import DesignProblem
from .result import DesignResult
from .opt_design_helpers import get_design_result, get_average_result_dict
from .change_dataframe import get_fim_addition, get_derivatives
import numpy as np
from typing import Iterable
from .task import ExpDesignSingleTask
from ..engine import Engine, MultiProcessEngine, SingleCoreEngine


def single_design_algo(design_result: DesignResult,
                       x: Iterable[float],
                       index: int
                       ) -> DesignResult:
    """
    Algorithm to find the single best condition to be added.
    This function is called once for every parameter set specified in
    design_problem.initial_x.
    For all candidates in design_problem.experiment_list measurements are
    simulated and used to compute the new FIM.
    It's eigenvalues are used to compute criteria values.
    The results are saved in 'design_result.single_runs'


    Parameters
    ----------
    design_result: result object for an experimental design setting
    x: the single set of parameters that will be used
    index: the index of the single x in the list design_problem.initial_x


    Returns
    -------
    design_result:
        the altered design_result
    """
    design_problem = design_result.design_problem

    # initialize as empty list
    dict_of_timepoints_cond = {key: [] for key in
                               design_problem.petab_problem.condition_df
                               .index.to_list()}

    # for each experimental conditions, save the time points at which we
    # consider a measurement for any candidate in experimental_list
    for candidate in design_problem.experiment_list:
        measurement_df = candidate['measurement_df']
        for ind, cond in enumerate(measurement_df.simulationConditionId):
            dict_of_timepoints_cond[cond].append(measurement_df.time[ind])

    # can the efficiency be  improved here?
    for cond in dict_of_timepoints_cond:
        dict_of_timepoints_cond[cond] = sorted(
            list(set(dict_of_timepoints_cond[cond])))

    # simulate all conditions forward once, save jacobian in a similar dict
    # as above
    deriv_dict = get_derivatives(design_problem,
                                 dict_of_timepoints_cond,
                                 x)

    this_single_run = []

    bad_sample = False
    for cand_ind, candidate in enumerate(design_problem.experiment_list):
        # we will have: new_FIM = old_FIM + A*A^T
        A = get_fim_addition(design_problem,
                             candidate,
                             deriv_dict,
                             dict_of_timepoints_cond)

        if np.isnan(A).any():
            bad_sample = True
            # TODO we don't have to append anything here, right?
            this_single_run.append(
                get_design_result(design_problem=design_problem,
                                  candidate=candidate,
                                  x=x,
                                  hess=None,
                                  initial_result=design_result.initial_result[
                                      index]))
            break
        else:
            this_single_run.append(
                get_design_result(design_problem=design_problem,
                                  candidate=candidate,
                                  x=x,
                                  hess=design_result.initial_result[index][
                                      'hess'],
                                  hess_additional=A,
                                  initial_result=design_result.initial_result[
                                      index]))

        this_single_run[cand_ind]['hess_additional'] = A

    # TODO rather bad indicator, change this to something clearer
    if bad_sample:
        this_single_run = 'bad'
    return this_single_run


def single_design_average(design_result: DesignResult):
    average_design_result = []

    for i in range(len(design_result.single_runs[0])):
        # only use the indices where no simulation failed ie no nan is anywhere
        list_of_dicts = [design_result.single_runs[good_i][i] for good_i
                         in design_result.good_initial_x_indices]

        average_design_result.append(get_average_result_dict(list_of_dicts))
    return average_design_result


def get_initial_results(design_result: DesignResult) -> DesignResult:
    # TODO decide how to handle the faster methods for some criteria here

    design_problem = design_result.design_problem

    design_result.initial_result = []
    for i, x in enumerate(design_problem.initial_x):
        init_res = get_design_result(design_problem=design_problem,
                                     candidate=None,
                                     x=x)
        # inverse of the FIM might be used for the computation of the det
        # TODO for faster det computation, handle case of FIM is not invertible
        try:
            init_res['hess_inv'] = np.linalg.inv(init_res['hess'])
        except np.linalg.LinAlgError:
            print("can't invert FIM for parameter set:", i)

        if design_problem.modified_criteria:
            try:
                hess_mod = init_res['hess'] + \
                           design_problem.const_for_hess * np.eye(
                    len(init_res['hess']))
                init_res['hess_inv_modified'] = np.linalg.inv(hess_mod)
            except TypeError:
                hess_mod = None
                init_res['hess_inv_modified'] = None

        # TODO can be made optional
        # eigenvalue decomposition (including eigenvectors as this is needed for
        # eigmin estimates
        try:
            init_res['eigen_decomposition'] = np.linalg.eigh(init_res['hess'])
        except np.linalg.LinAlgError:
            init_res['eigen_decomposition'] = None
        design_result.initial_result.append(init_res)

    return design_result


def run_exp_design(design_problem: DesignProblem,
                   engine: Engine = None) -> DesignResult:
    """
    The main method for experimental design.

    Parameters
    ----------
    design_problem:
        the  problem formulation for experimental design
    engine: Parallelization engine. Defaults to sequential execution on a
        SingleCoreEngine.
        (only small parts are parallelized so far)

    Returns
    -------
    design_result:
        the result object which contains criteria values for each candidate
        condition which is to be tested
    """
    design_result = DesignResult(design_problem=design_problem)

    # if only one x is passed, convert it into a list of lists
    if not all(isinstance(elem, Iterable) for elem in
               design_problem.initial_x):
        design_problem.initial_x = [design_problem.initial_x]

    design_result = get_initial_results(design_result)

    # compute the results for each candidate specified in
    # design_problem.experimental_list for each x in design_problem.initial_x
    # save result in a list in design_result.single_runs

    tasks = []
    for i, x in enumerate(design_problem.initial_x):
        task = ExpDesignSingleTask(design_result=design_result,
                                   x=x,
                                   index=i)
        tasks.append(task)

    # if engine is None:
    #     engine = SingleCoreEngine()
    engine = MultiProcessEngine()

    design_result.single_runs = engine.execute(tasks)

    print("Finished single_runs")

    # the simulation may fail for some parameter sets, in this case we
    # exclude them completely
    bad = []
    for ind, res in enumerate(design_result.single_runs):
        if res == 'bad':
            bad.append(ind)

    if bad:
        print("The simulation with the parameters at the following indices "
              "from design_problem.initial_x failed and will be excluded in "
              "the computations: ", bad)
    design_result.good_initial_x_indices = [i for i in range(len(
        design_problem.initial_x)) if i not in bad]

    # if multiple sets of parameters were passed, compute average values for
    # the criteria that will be used later when checking combinations
    if len(design_problem.initial_x) > 1:
        design_result.single_runs_average = single_design_average(
            design_result=design_result)
    return design_result
