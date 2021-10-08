"""
Visualization of the model fit after optimization.

Currently only for PEtab problems.
"""
import petab
from amici.petab_simulate import simulate_petab
import amici.petab_import as petab_import
from petab.visualize import plot_problem
from typing import Union, Sequence
from ..result import Result
from amici.petab_objective import rdatas_to_simulation_df


def visualize_optimized_model_fit(petab_problem: petab.Problem,
                                  result: Union[Result, Sequence[Result]],
                                  **kwargs
                                  ):
    """
    Visualize the optimized model fit of a PEtab problem.
    Function calls the PEtab visualization file of the petab_problem and
    visualizes the fit of the optimized parameter. Common additional
    argument is `subplot_dir` to specify the directory each subplot is
    saved to. Further keyword arguments are delegated to
    petab.visualize.plot_with_vis_spec, see there for more information.

    Parameters
    ----------
    petab_problem:
        The :py:class:`petab.Problem` that was optimized.
    result:
        The result object from optimization.

    Returns
    -------
    ax: Axis object of the created plot.
    None: In case subplots are saved to file
    """
    problem_parameters = \
        dict(zip(petab_problem.parameter_df.index,
                 result.optimize_result.list[0]['x']))

    amici_model = petab_import.import_petab_problem(
        petab_problem,
        model_output_dir=kwargs.pop('model_output_dir'))

    res = simulate_petab(petab_problem, amici_model=amici_model,
                         scaled_parameters=True,
                         problem_parameters=problem_parameters,
                         solver=kwargs.pop('amici_solver', None))

    sim_df = rdatas_to_simulation_df(res["rdatas"],
                                     amici_model,
                                     petab_problem.measurement_df)

    # function to call, to plot data and simulations
    ax = plot_problem(petab_problem=petab_problem,
                      simulations_df=sim_df,
                      **kwargs
                      )
    return ax
