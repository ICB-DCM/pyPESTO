import petab
from amici.petab_simulate import simulate_petab
import amici.petab_import as petab_import
from petab.visualize import plot_with_vis_spec
from typing import Union, Sequence
from ..result import Result
from amici.petab_objective import rdatas_to_simulation_df


def visualize_optimized_model_fit(petab_problem: petab.Problem,
                                  result: Union[Result, Sequence[Result]],
                                  **kwargs
                                  ):
    """
    Function calls the petab.visualize file of the petab_problem and
    visualizes the fit of the optimized parameter. Possible additional
    argument is `subplot_dir` to specify the directory each subplot is
    saved to.

    Parameters
    ----------
    petab_problem:
        The petab_problem that was optimized.
    result:
        The result object from optimization.

    Returns
    -------
    ax: Axis object of the created plot.
    None: In case subplots are save to file
    """
    problem_parameters = \
        dict(zip(petab_problem.parameter_df.index,
                 result.optimize_result.list[0]['x']))

    amici_model = petab_import.import_petab_problem(petab_problem)

    res = simulate_petab(petab_problem, amici_model=amici_model,
                         scaled_parameters=True,
                         problem_parameters=problem_parameters)

    sim_df = rdatas_to_simulation_df(res["rdatas"],
                                     amici_model,
                                     petab_problem.measurement_df)

    # function to call, to plot data and simulations
    ax = plot_with_vis_spec(vis_spec_df=petab_problem.visualization_df,
                            conditions_df=petab_problem.condition_df,
                            measurements_df=petab_problem.measurement_df,
                            simulations_df=sim_df,
                            **kwargs
                            )
    return ax
