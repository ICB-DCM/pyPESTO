import numpy as np
from .design_problem import DesignProblem
from petab.C import OBSERVABLE_ID, TIME, NOISE_PARAMETERS, \
    MEASUREMENT, SIMULATION_CONDITION_ID, NORMAL, LAPLACE
import amici
import petab
from .opt_design_helpers import update_pypesto_from_petab
from amici.petab_objective import simulate_petab
from amici.petab_objective import rdatas_to_simulation_df


# TODO how to choose settings here?
def simulate_forward(petab_problem: petab.Problem,
                     x: np.ndarray,
                     model: amici.ModelPtr,
                     condition_name: str,
                     last_timepoint: float,
                     candidate: dict) -> amici.ReturnDataView:
    """
    performs a forward simulation with x as parameter and condition
    'condition_name' until 'last_timepoint'
    looks up the names of fixed parameters of the model, goes into
    petab_problem.condition_df and sets the values there for the condition
    'condition_name' as fixed parameters of the model for the simulation

    Parameters
    ----------
    petab_problem:
        the petab problem
    x:
        the parameter to be used for the simulation
    model:
        the amici model
    condition_name:
        the name of the condition for which we want to simulate
    last_timepoint:
        the time point we are interested in, we will simulate until that time

    Returns
    -------
    rdata:
        amici.ReturnDataView which in particular contains the values at
        last_timepoint
    """
    solver = model.getSolver()
    # solver.setSensitivityMethod(amici.SensitivityMethod_forward)
    # solver.setSensitivityOrder(amici.SensitivityOrder_first)

    # max is for timepoints<1
    model.setTimepoints(amici.DoubleVector(
        np.linspace(0, last_timepoint, max(int(last_timepoint + 1), 2))))
    # find entries for conditions and map correctly
    all_fixed_params = []

    for fixed_par in model.getFixedParameterIds():
        if 'condition_df' in candidate:
            all_fixed_params.append(
                candidate['condition_df'][fixed_par].loc[condition_name])
        else:
            all_fixed_params.append(
                petab_problem.condition_df[fixed_par].loc[condition_name])
    model.setFixedParameters(amici.DoubleVector(np.array(all_fixed_params)))
    model.setParameterScale(amici.ParameterScaling_log10)
    temp = x
    # these may be noise parameters for which we have explicit values
    missing_params = len(model.getParameterIds()) - len(temp)
    for i in range(missing_params):
        temp = np.append(temp, 0)
    model.setParameters(amici.DoubleVector(temp))

    rdata = amici.runAmiciSimulation(model, solver, None)
    return rdata


def add_candidate_to_dfs(design_problem: DesignProblem, candidate: dict,
                         x: np.ndarray = None) \
        -> DesignProblem:
    """
    changes design_problem.petab_problem
    adds new rows to measurement_df
    the new measurement in measurement_df is computed via forward simulation
    with x as parameter
    noise is added
    """

    # TODO this only works with fixed numbers as noise not with estimated
    # parameters right now

    if x is None:
        x = design_problem.initial_x

    # add new row to measurement df
    measurement_df = design_problem.petab_problem.measurement_df
    measurement_df = measurement_df.append(candidate['measurement_df'],
                                           ignore_index=True)
    design_problem.petab_problem.measurement_df = measurement_df

    # using 'simulate_petab'
    # is slower than other method
    # design_problem = write_measurement_alternative(
    # design_problem=design_problem, x=x)

    # using explicit simulation by hand
    design_problem = write_measurement(design_problem=design_problem,
                                       candidate=candidate,
                                       x=x)

    return design_problem


def write_measurement(design_problem: DesignProblem, candidate: dict,
                      x: np.ndarray = None) -> DesignProblem:
    if x is None:
        x = design_problem.initial_x

    measurement_df = design_problem.petab_problem.measurement_df

    for row_index in range(len(candidate['measurement_df'])):
        measurement_time = candidate['measurement_df'][TIME][row_index]
        condition_name = candidate['measurement_df'][SIMULATION_CONDITION_ID][
            row_index]
        rdata = simulate_forward(
            petab_problem=design_problem.petab_problem,
            x=x,
            model=design_problem.model,
            condition_name=condition_name,
            last_timepoint=measurement_time,
            candidate=candidate)

        have_noise = NOISE_PARAMETERS in design_problem.petab_problem. \
            measurement_df.columns

        if have_noise:
            noise = candidate['measurement_df'][NOISE_PARAMETERS][row_index]
            if isinstance(noise, (int, float)):
                expdata = amici.ExpData(rdata, noise, 0)
            elif isinstance(noise, str):
                raise NotImplementedError(
                    "please enter a fixed number for the error")
                # how to make connection between the name of the parameter
                # and the internal name for the noise here we would need
                # something like "noiseParameter1_observable"
                # since these are the names in ParameterIds()
                # noise_index = design_problem.model.getParameterIds(
                # ).index(noise)
                # expdata = amici.ExpData(rdata, x[noise_index], 0)
            else:
                raise Exception(
                    "noise in measurement table has to be a number or the "
                    "name of an estimated parameter")

        time_index = list(rdata['t']).index(measurement_time)
        observable_str = candidate['measurement_df'][OBSERVABLE_ID][row_index]
        observable_index = design_problem.model.getObservableIds().index(
            observable_str)

        # exp_data.getObservedData is a long list of all observables at all
        # time points, hence the indexing
        index = time_index * len(
            design_problem.model.getObservableIds()) + observable_index
        if have_noise:
            measurement_with_noise = expdata.getObservedData()[index]
        else:
            measurement_with_noise = rdata['y'][index][0]

        measurement_df[MEASUREMENT][len(measurement_df) - len(
            candidate['measurement_df']) + row_index] = measurement_with_noise

    # save into problem
    design_problem.petab_problem.measurement_df = measurement_df

    return design_problem


def delete_candidate_from_dfs(design_problem: DesignProblem, candidate: dict) \
        -> DesignProblem:
    """
    delete the new rows which where temporarily added to the measurement
    dataframe
    """

    # delete the measurement row
    # use condition_id as unique identifier
    measurement_df = design_problem.petab_problem.measurement_df

    id_to_be_deleted = []
    for row_index in range(len(candidate['measurement_df'])):
        cond_id = measurement_df[SIMULATION_CONDITION_ID][
            len(measurement_df) - len(candidate['measurement_df']) + row_index]
        id_to_be_deleted.append(measurement_df[SIMULATION_CONDITION_ID][
                                    measurement_df[
                                        SIMULATION_CONDITION_ID] ==
                                    cond_id].index.tolist())
    flat_list = [item for sublist in id_to_be_deleted for item in sublist]
    measurement_df = measurement_df.drop(flat_list)
    design_problem.petab_problem.measurement_df = measurement_df

    return design_problem

# alternative to get the new measurement value
# is slower than the other one

# def write_measurement_alternative(design_problem: DesignProblem,
#                                   x: np.ndarray = None) -> DesignProblem:
#     """
#     takes design_problem.measurement_df and searches for all rows where the
#     measurement is still float('NaN').
#     Simulates the measurement value for this and adds normal or laplace noise
#
#     Parameters
#     ----------
#     design_problem
#     x
#
#     Returns
#     -------
#
#     """
#     if x is None:
#         x = design_problem.x
#
#     simulation_conditions = design_problem.petab_problem \
#         .get_simulation_conditions_from_measurement_df()
#
#     dict_x = {name: x[i] for i, name in
#               enumerate(design_problem.petab_problem.parameter_df.index)}
#     rdata_sim_petab = simulate_petab(
#         petab_problem=design_problem.petab_problem,
#         amici_model=design_problem.model,
#         simulation_conditions=simulation_conditions,
#         problem_parameters=dict_x,
#         scaled_parameters=True)
#
#     measurement_df = design_problem.petab_problem.measurement_df
#     observable_df = design_problem.petab_problem.observable_df
#
#     sim_df = rdatas_to_simulation_df(rdatas=rdata_sim_petab['rdatas'],
#                                      model=design_problem.model,
#                                      measurement_df=measurement_df)
#     bool_array = pd.isna(measurement_df.measurement)
#     indices_to_replace = np.where(bool_array)[0]
#
#     for index in indices_to_replace:
#         measurement_df.measurement[index] = sim_df.simulation[index]
#         # add noise
#         noise_distributions_dict = amici.petab_import.get_observation_model(
#             observable_df)[1]
#         observable = measurement_df[OBSERVABLE_ID][index]
#         noise_distribution = noise_distributions_dict[observable]
#
#         noise_value = measurement_df[NOISE_PARAMETERS][index]
#         # TODO log scale ?
#         if noise_distribution == NORMAL or noise_distribution ==
#         'lin-normal':
#             measurement_df.measurement[index] = measurement_df.measurement[
#                                                     index] +
#                                                     np.random.normal(
#                 0, noise_value)
#         elif noise_distribution == LAPLACE or noise_distribution == \
#                 'lin-laplace':
#             measurement_df.measurement[index] = measurement_df.measurement[
#                                                     index] +
#                                                     np.random.laplace(
#                 0, noise_value)
#         else:
#             raise ValueError("Unknown noise distribution. Only 'normal' and "
#                              "'laplace' are supported")
#
#     # save into problem
#     design_problem.petab_problem.measurement_df = measurement_df
#     return design_problem


# def simulate(design_problem: DesignProblem, x: np.ndarray) \
#         -> List[amici.ReturnDataView]:
#     # update objective, otherwise 'ret' doesn't contain the new condition
#     design_problem = update_pypesto_from_petab(design_problem)
#     ret = design_problem.problem.objective(x, return_dict=True)
#     return ret['rdatas']
