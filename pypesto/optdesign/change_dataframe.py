import numpy as np
from amici.petab_objective import simulate_petab
from amici.petab_objective import rdatas_to_simulation_df
from .opt_design_helpers import update_pypesto_from_petab
from .design_problem import DesignProblem
from ..petab import PetabImporter
from typing import List
import amici


# TODO choosing of time points is arbitrary, switch to the other method for simulating forward
def simulate_forward(x: np.ndarray, model: amici.ModelPtr, condition_entries: List[float],
                     last_timepoint: float) -> amici.ReturnDataView:
    solver = model.getSolver()
    solver.setSensitivityMethod(amici.SensitivityMethod_forward)
    solver.setSensitivityOrder(amici.SensitivityOrder_first)

    # max is for timepoints<1
    model.setTimepoints(amici.DoubleVector(np.linspace(0, last_timepoint, max(int(last_timepoint + 1), 2))))
    model.setFixedParameters(condition_entries)
    model.setParameterScale(amici.ParameterScaling_log10)
    temp = x
    # these may be noise parameters for which we have explicit values
    missing_params = len(model.getParameterIds()) - len(temp)
    for i in range(missing_params):
        temp = np.append(temp, 0)
    model.setParameters(temp)

    rdata = amici.runAmiciSimulation(model, solver, None)
    return rdata


def simulate(design_problem: DesignProblem, x: np.ndarray) -> List[amici.ReturnDataView]:
    # update objective, otherwise 'ret' doesn't contain the new condition
    design_problem = update_pypesto_from_petab(design_problem)
    ret = design_problem.problem.objective(x, return_dict=True)
    return ret['rdatas']


def add_candidate_to_dfs(design_problem: DesignProblem, candidate: dict, x: np.ndarray) \
        -> (DesignProblem, float):
    """
    changes design_problem.petab_problem
    adds a new row to condition_df, measurement_df and if specified observable_df
    the new measurement in measurement_df is computed via forward simulation with x as parameter
    noise is added
    """

    # TODO this only works with fixed numbers as noise not with estimated parameters right now

    # add new row to observable df
    if candidate['observable_df'] is not None:
        observable_df = design_problem.petab_problem.observable_df.reset_index()
        observable_df.loc[len(observable_df)] = candidate['observable_df']
        observable_df = observable_df.set_index('observableId')
        design_problem.petab_problem.observable_df = observable_df

    # doesn't do anything at the moment. because of the way we simulate forward
    # may still be useful
    # add new row to condition df
    condition_df = design_problem.petab_problem.condition_df.reset_index()
    condition_df.loc[len(condition_df)] = candidate['condition_df']
    condition_df = condition_df.set_index('conditionId')
    design_problem.petab_problem.condition_df = condition_df

    # add new row to measurement df
    measurement_df = design_problem.petab_problem.measurement_df
    measurement_df.loc[len(measurement_df)] = candidate['measurement_df']
    design_problem.petab_problem.measurement_df = measurement_df

    # TODO change the way we simulate data / add noise to make it more general
    """
    # method using rdatas_to..
    rdatas_from_ret = simulate(design_problem=design_problem, x=x)
    importer = PetabImporter(design_problem.petab_problem, model_name=design_problem.model.getName())
    """

    # write correct new measurement into the new row
    index_for_time_in_df = design_problem.petab_problem.measurement_df.columns.get_loc("time")
    measurement_time = candidate['measurement_df'][index_for_time_in_df]

    # candidate['condition_df'][2:] since we skip the names for the condition
    rdata = simulate_forward(x=design_problem.result.optimize_result.as_list(['x'])[0]['x'],
                             model=design_problem.model,
                             condition_entries=candidate['condition_df'][2:],
                             last_timepoint=measurement_time)

    # TODO i hope this chooses the correct noise model
    have_noise = "noiseParameters" in design_problem.petab_problem.measurement_df.columns

    if have_noise:
        index_for_sd_in_df = design_problem.petab_problem.measurement_df.columns.get_loc("noiseParameters")
        noise = candidate['measurement_df'][index_for_sd_in_df]
        if isinstance(noise, float):
            expdata = amici.ExpData(rdata, noise, 0)
        elif isinstance(noise, str):
            raise NotImplementedError("please enter a fixed number for the error")
            # how to make connection between the name of the parameter and the internal name for the noise
            # here we would need something like "noiseParameter1_observable" since these are the names in ParameterIds()
            # noise_index = design_problem.model.getParameterIds().index(noise)
            # expdata = amici.ExpData(rdata, x[noise_index], 0)
        else:
            raise Exception("noise in measurement table has to be float or the name of an estimated parameter")

    # TODO i don't know how to create expData without adding noise...
    time_index = list(rdata['t']).index(measurement_time)
    index_for_obs_id = design_problem.petab_problem.measurement_df.columns.get_loc("observableId")
    observable_str = candidate['measurement_df'][index_for_obs_id]
    observable_index = design_problem.model.getObservableIds().index(observable_str)

    # exp_data.getObservedData is a long list of all observables at all time points, hence the indexing
    index = time_index * len(design_problem.model.getObservableIds()) + observable_index
    if have_noise:
        measurement_with_noise = expdata.getObservedData()[index]
    else:
        measurement_with_noise = rdata['y'][index][0]
    measurement_df.at[len(measurement_df) - 1, 'measurement'] = measurement_with_noise

    # save into problem
    design_problem.petab_problem.measurement_df = measurement_df

    return design_problem, measurement_with_noise


def delete_candidate_from_dfs(design_problem: DesignProblem, candidate: dict):
    """
    delete the new rows which where temporarily added to the observable, condition and measurement dataframe
    """
    # delete the observable row
    # use observableID as unique identifier
    if candidate['observable_df'] is not None:
        # the observable_id should always be the first entry
        observable_name = candidate['observable_df'][0]
        observable_df = design_problem.petab_problem.observable_df
        observable_df = observable_df.drop(observable_name)
        design_problem.petab_problem.observable_df = observable_df

    # delete the condition row
    # the condition_id should always be the first entry
    condition_name = candidate['condition_df'][0]
    condition_df = design_problem.petab_problem.condition_df
    condition_df = condition_df.drop(condition_name)
    design_problem.petab_problem.condition_df = condition_df

    # delete the measurement row
    # use condition_id as unique identifier
    index_sim_cond = design_problem.petab_problem.measurement_df.columns.get_loc("simulationConditionId")
    measurement_df = design_problem.petab_problem.measurement_df
    test = measurement_df['simulationConditionId'][
        measurement_df['simulationConditionId'] == candidate['measurement_df'][index_sim_cond]].index.tolist()
    measurement_df = measurement_df.drop(test)
    design_problem.petab_problem.measurement_df = measurement_df

    return design_problem
