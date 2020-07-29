import numpy as np
from .opt_design_helpers import update_pypesto_from_petab
from .design_problem import DesignProblem
from typing import List
import amici


# TODO choosing of time points is arbitrary, switch to the other method for
#  simulating forward
def simulate_forward(x: np.ndarray, model: amici.ModelPtr,
                     condition_entries: List[float],
                     last_timepoint: float) -> amici.ReturnDataView:
    solver = model.getSolver()
    solver.setSensitivityMethod(amici.SensitivityMethod_forward)
    solver.setSensitivityOrder(amici.SensitivityOrder_first)

    # max is for timepoints<1
    model.setTimepoints(amici.DoubleVector(
        np.linspace(0, last_timepoint, max(int(last_timepoint + 1), 2))))
    model.setFixedParameters(amici.DoubleVector(condition_entries))
    model.setParameterScale(amici.ParameterScaling_log10)
    temp = x
    # these may be noise parameters for which we have explicit values
    missing_params = len(model.getParameterIds()) - len(temp)
    for i in range(missing_params):
        temp = np.append(temp, 0)
    model.setParameters(amici.DoubleVector(temp))

    rdata = amici.runAmiciSimulation(model, solver, None)
    return rdata


def simulate(design_problem: DesignProblem, x: np.ndarray) \
        -> List[amici.ReturnDataView]:
    # update objective, otherwise 'ret' doesn't contain the new condition
    design_problem = update_pypesto_from_petab(design_problem)
    ret = design_problem.problem.objective(x, return_dict=True)
    return ret['rdatas']


def add_candidate_to_dfs(design_problem: DesignProblem, candidate: dict,
                         x: np.ndarray) \
        -> DesignProblem:
    """
    changes design_problem.petab_problem
    adds a new row to condition_df, measurement_df and if specified
    observable_df
    the new measurement in measurement_df is computed via forward simulation
    with x as parameter
    noise is added
    """

    # TODO this only works with fixed numbers as noise not with estimated
    #  parameters right now

    # add new row to observable df
    if candidate['observable_df'] is not None:
        observable_df = design_problem.petab_problem.observable_df \
            .reset_index()
        observable_df = observable_df.append(
            candidate['observable_df'].reset_index())
        observable_df = observable_df.set_index('observableId')
        design_problem.petab_problem.observable_df = observable_df

    # doesn't do anything at the moment. because of the way we simulate forward
    # may still be useful
    # add new row to condition df
    condition_df = design_problem.petab_problem.condition_df.reset_index()
    condition_df = condition_df.append(
        candidate['condition_df'].reset_index())
    condition_df = condition_df.set_index('conditionId')
    design_problem.petab_problem.condition_df = condition_df

    # add new row to measurement df
    measurement_df = design_problem.petab_problem.measurement_df
    measurement_df = measurement_df.append(candidate['measurement_df'],
                                           ignore_index=True)
    design_problem.petab_problem.measurement_df = measurement_df

    # TODO change the way we simulate data / add noise to make it more general
    """
    from amici.petab_objective import simulate_petab
    from amici.petab_objective import rdatas_to_simulation_df
    from ..petab import PetabImporter

    # method using rdatas_to..
    rdatas_from_ret = simulate(design_problem=design_problem, x=x)
    importer = PetabImporter(design_problem.petab_problem,
                             model_name=design_problem.model.getName())
    """

    # write correct new measurement into the new row
    for row_index in range(len(candidate['measurement_df'])):
        measurement_time = candidate['measurement_df']['time'][row_index]

        # candidate['condition_df'][2:] since we skip the names for the
        # condition
        rdata = simulate_forward(
            x=design_problem.result.optimize_result.as_list(['x'])[0]['x'],
            model=design_problem.model,
            condition_entries=candidate['condition_df'][2:],
            last_timepoint=measurement_time)

        # TODO i hope this chooses the correct noise model
        have_noise = "noiseParameters" in \
                     design_problem.petab_problem.measurement_df.columns

        if have_noise:
            noise = candidate['measurement_df']['noiseParameters'][row_index]
            if isinstance(noise, float):
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
                    "noise in measurement table has to be float or the name "
                    "of an estimated parameter")

        # TODO i don't know how to create expData without adding noise...
        time_index = list(rdata['t']).index(measurement_time)
        observable_str = candidate['measurement_df']["observableId"][row_index]
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

        measurement_df['measurement'][len(measurement_df) - len(
            candidate['measurement_df']) + row_index] = measurement_with_noise

    # save into problem
    design_problem.petab_problem.measurement_df = measurement_df

    return design_problem


def delete_candidate_from_dfs(design_problem: DesignProblem, candidate: dict):
    """
    delete the new rows which where temporarily added to the observable,
    condition and measurement dataframe
    """
    # delete the observable row
    # use observableID as unique identifier
    if candidate['observable_df'] is not None:
        observable_names = candidate['observable_df'].index
        observable_df = design_problem.petab_problem.observable_df
        observable_df = observable_df.drop(observable_names)
        design_problem.petab_problem.observable_df = observable_df

    # delete the condition row
    # the condition_id should always be the first entry
    condition_names = candidate['condition_df'].index
    condition_df = design_problem.petab_problem.condition_df
    condition_df = condition_df.drop(condition_names)
    design_problem.petab_problem.condition_df = condition_df

    # delete the measurement row
    # use condition_id as unique identifier
    measurement_df = design_problem.petab_problem.measurement_df

    id_to_be_deleted = []
    for row_index in range(len(candidate['measurement_df'])):
        cond_id = measurement_df['simulationConditionId'][
            len(measurement_df) - len(candidate['measurement_df']) + row_index]
        id_to_be_deleted.append(measurement_df['simulationConditionId'][
            measurement_df['simulationConditionId'] == cond_id].index.tolist())
    flat_list = [item for sublist in id_to_be_deleted for item in sublist]
    measurement_df = measurement_df.drop(flat_list)
    design_problem.petab_problem.measurement_df = measurement_df

    return design_problem
