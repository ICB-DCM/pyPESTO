from typing import Dict, List, Sequence, Union
import copy

from ..objective.amici_calculator import (
    AmiciCalculator, calculate_function_values)
from ..objective.amici_util import (
    get_error_output
)
from .problem import InnerProblem
from .solver import InnerSolver, AnalyticalInnerSolver, NumericalInnerSolver

try:
    import amici
    from amici.parameter_mapping import ParameterMapping
except ImportError:
    amici = None

AmiciModel = Union['amici.Model', 'amici.ModelPtr']
AmiciSolver = Union['amici.Solver', 'amici.SolverPtr']


class HierarchicalAmiciCalculator(AmiciCalculator):
    """
    A calculator is passed as `calculator` to the pypesto.AmiciObjective.
    While this class cannot be used directly, it has two subclasses
    which allow to use forward or adjoint sensitivity analysis to
    solve a `pypesto.HierarchicalProblem` efficiently in an inner loop,
    while the outer optimization is only concerned with variables not
    specified as `pypesto.HierarchicalParameter`s.
    """

    def __init__(self,
                 inner_problem: InnerProblem,
                 inner_solver: InnerSolver = None):
        """
        Initialize the calculator from the given problem.
        """
        self.inner_problem = inner_problem

        if inner_solver is None:
            #inner_solver = NumericalInnerSolver()
            inner_solver = AnalyticalInnerSolver()
        self.inner_solver = inner_solver

    def __call__(self,
                 x_dct: Dict,
                 sensi_order: int,
                 mode: str,
                 amici_model: AmiciModel,
                 amici_solver: AmiciSolver,
                 edatas: List['amici.ExpData'],
                 n_threads: int,
                 x_ids: Sequence[str],
                 parameter_mapping: 'ParameterMapping'):

        dim = len(x_ids)

        # set order in solver to 0
        amici_solver.setSensitivityOrder(0)

        # fill in boring values
        x_dct = copy.deepcopy(x_dct)
        for key, val in self.inner_problem.get_boring_pars(
                scaled=True).items():
            x_dct[key] = val

        # fill in parameters
        # TODO (#226) use plist to compute only required derivatives
        amici.parameter_mapping.fill_in_parameters(
            edatas=edatas,
            problem_parameters=x_dct,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=amici_model
        )

        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            amici_model,
            amici_solver,
            edatas,
            num_threads=min(n_threads, len(edatas)),
        )

        # check if any simulation failed
        if any([rdata['status'] < 0.0 for rdata in rdatas]):
            return get_error_output(amici_model, edatas, rdatas, dim)

        sim = [rdata['y'] for rdata in rdatas]
        sigma = [rdata['sigmay'] for rdata in rdatas]

        # compute optimal inner parameters
        x_inner_opt = self.inner_solver.solve(
            self.inner_problem, sim, sigma, scaled=True)
        #print(x_inner_opt)
        # fill in optimal values
        x_dct = copy.deepcopy(x_dct)
        for key, val in x_inner_opt.items():
            x_dct[key] = val

        # fill in parameters
        # TODO (#226) use plist to compute only required derivatives
        amici.parameter_mapping.fill_in_parameters(
            edatas=edatas,
            problem_parameters=x_dct,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=amici_model
        )

        amici_solver.setSensitivityOrder(sensi_order)

        # resimulate
        # run amici simulation
        rdatas = amici.runAmiciSimulations(
            amici_model,
            amici_solver,
            edatas,
            num_threads=min(n_threads, len(edatas)),
        )

        return calculate_function_values(
            rdatas, sensi_order, mode, amici_model, amici_solver, edatas,
            x_ids, parameter_mapping)
