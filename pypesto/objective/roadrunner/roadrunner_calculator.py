import copy
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import petab
import roadrunner
from petab.parameter_mapping import ParMappingDictQuadruple

from ...C import MODE_FUN, MODE_RES, ModeType
from .utils import ExpData, unscale_parameters


class RoadRunnerCalculator:
    """Class to handle RoadRunner simulation and obtain objective value."""

    def __init__(self):
        pass

    def __call__(
        self,
        x_dct: Dict,  # TODO: sensi_order support
        mode: ModeType,
        roadrunner_instance: roadrunner.RoadRunner,
        edatas: List[ExpData],
        x_ids: Sequence[str],
        parameter_mapping: List[ParMappingDictQuadruple],
        petab_problem: petab.Problem,
    ):
        """Perform the RoadRunner call and obtain objective function values.

        Parameters
        ----------
        x_dct:
            Parameter dictionary.
        sensi_orders:
            Tuple of sensitivity orders.
        mode:
            Mode of the call.
        roadrunner_instance:
            RoadRunner instance.
        edatas:
            List of ExpData.
        x_ids:
            Sequence of parameter IDs.
        parameter_mapping:
            Parameter parameter_mapping.

        Returns
        -------
        Tuple of objective function values.
        """
        # sanitiy check that edatas and conditions are consistent
        if len(edatas) != len(parameter_mapping):
            raise ValueError(
                "Number of edatas and conditions are not consistent."
            )
        simulation_results = []
        for edata, mapping_per_condition in zip(edatas, parameter_mapping):
            sim_res = self.simulate_per_condition(
                x_dct, roadrunner_instance, edata, mapping_per_condition
            )
            # fill a corresponding dataframe with the simulation results
            sim_res_df = self.fill_simulation_df(sim_res, edata)
            simulation_results.append(sim_res_df)
        simulation_results = pd.concat(simulation_results)

        if mode == MODE_FUN:
            llh = petab.calculate_llh(
                petab_problem.measurement_df,
                simulation_results,
                petab_problem.observable_df,
                petab_problem.parameter_df,
            )
            return {"fval": -llh, "simulation_results": simulation_results}
        if mode == MODE_RES:
            res_df = petab.calculate_residuals(
                petab_problem.measurement_df,
                simulation_results,
                petab_problem.observable_df,
                petab_problem.parameter_df,
            )
            return {"res": res_df, "simulation_results": simulation_results}

    def simulate_per_condition(
        self,
        x_dct: Dict,
        roadrunner_instance: roadrunner.RoadRunner,
        edata: ExpData,
        parameter_mapping_per_condition: ParMappingDictQuadruple,
    ):
        """Simulate the model for a single condition.

        Parameters
        ----------
        x_dct:
            Parameter dictionary.
        roadrunner_instance:
            RoadRunner instance.
        edata:
            ExpData of a single condition.
        parameter_mapping_per_condition:
            Parameter parameter_mapping for a single condition.
        """
        # get timepoints and outputs to simulate
        timepoints = edata.get_timepoints()
        # Convert integers to floats
        timepoints = list(map(float, timepoints))
        if timepoints[0] != 0.0:
            timepoints = [0.0] + timepoints
        if len(timepoints) == 1:
            timepoints = [0.0] + timepoints
        observables_ids = edata.get_observable_ids()
        # steady state stuff
        steady_state_calculations = False
        state_variables = roadrunner_instance.model.getFloatingSpeciesIds()
        obs_ss = []
        state_ss = []

        # if the first and third parameter mappings are not empty, we need
        # to pre-equlibrate the model
        if (
            parameter_mapping_per_condition[0]
            and parameter_mapping_per_condition[2]
        ):
            steady_state_calculations = True
            roadrunner_instance.conservedMoietyAnalysis = True
            self.fill_in_parameters(
                x_dct,
                roadrunner_instance,
                parameter_mapping_per_condition,
                preeq=True,
            )
            # steady state output = observables + state variables
            steady_state_selections = observables_ids + state_variables
            roadrunner_instance.steadyStateSelections = steady_state_selections
            steady_state = roadrunner_instance.getSteadyStateValuesNamedArray()
            # we split the steady state into observables and state variables
            obs_ss = steady_state[:, : len(observables_ids)].flatten()
            state_ss = steady_state[:, len(observables_ids) :].flatten()
            # turn off conserved moiety analysis
            roadrunner_instance.conservedMoietyAnalysis = False
            # reset the model
            roadrunner_instance.reset()
        # set parameters
        self.fill_in_parameters(
            x_dct, roadrunner_instance, parameter_mapping_per_condition
        )
        # if steady state calculations are required, set state variables
        if steady_state_calculations:
            roadrunner_instance.setValues(state_variables, state_ss)
            # fill in overriden species
            self.fill_in_parameters(
                x_dct,
                roadrunner_instance,
                parameter_mapping_per_condition,
                filling_mode="only_species",
            )

        sim_res = roadrunner_instance.simulate(
            times=timepoints, selections=["time"] + observables_ids
        )

        # reset the model
        roadrunner_instance.reset()

        return sim_res

    def fill_in_parameters(
        self,
        problem_parameters: Dict,
        roadrunner_instance: Optional[roadrunner.RoadRunner] = None,
        parameter_mapping: Optional[ParMappingDictQuadruple] = None,
        preeq: bool = False,
        filling_mode: Optional[str] = None,
    ):
        """Fill in parameters into the roadrunner instance.

        Largly taken from amici.petab.parameter_mapping.fill_in_parameters

        Parameters
        ----------
        roadrunner_instance:
            RoadRunner instance to fill in parameters
        problem_parameters:
            Problem parameters as parameterId=>value dict. Only
            parameters included here will be set. Remaining parameters will
            be used as already set in `amici_model` and `edata`.
        parameter_mapping:
            Parameter mapping for current condition.
        preeq:
            Whether to fill in parameters for pre-equilibration.
        filling_mode:
            Which parameters to fill in. If None or "all",
            all parameters are filled in.
            Other options are "only_parameters" and "only_species".
        """
        if filling_mode is None:
            filling_mode = "all"
        mapping = parameter_mapping[1]
        scaling = parameter_mapping[3]
        if preeq:
            mapping = parameter_mapping[0]
            scaling = parameter_mapping[2]
        # create a deepcopy of the mapping for comparison later
        mapping_orig = copy.deepcopy(mapping)

        # Parameter parameter_mapping may contain parameter_ids as values,
        # these *must* be replaced

        def _get_par(model_par, value, mapping):
            """Get parameter value from problem_parameters and mapping.

            Replace parameter IDs in parameter_mapping dicts by values from
            problem_parameters where necessary
            """
            if isinstance(value, str):
                try:
                    # estimated parameter
                    return problem_parameters[value]
                except KeyError:
                    # condition table overrides must have been handled already,
                    # e.g. by the PEtab parameter parameter_mapping, but
                    # parameters from InitialAssignments may still be present.
                    if mapping[value] == model_par:
                        # prevent infinite recursion
                        raise
                    return _get_par(value, mapping[value], mapping)
            if model_par in problem_parameters:
                # user-provided
                return problem_parameters[model_par]
            # prevent nan-propagation in derivative
            if np.isnan(value):
                return 0.0
            # constant value
            return value

        mapping_values = {
            key: _get_par(key, val, mapping) for key, val in mapping.items()
        }
        # we assume the parameters to be given in the scale defined in the
        # petab problem. Thus, they need to be unscaled.
        mapping_values = unscale_parameters(mapping_values, scaling)
        # seperate the parameters into ones that overwrite species and others
        mapping_params = dict()
        mapping_species = dict()
        for key, value in mapping_values.items():
            if key in roadrunner_instance.model.getFloatingSpeciesIds():
                # values that originally were NaN are not set
                if isinstance(mapping[key], str) or not np.isnan(mapping[key]):
                    mapping_species[key] = float(value)
            else:
                mapping_params[key] = value

        if filling_mode == "only_parameters" or filling_mode == "all":
            # set parameters.
            roadrunner_instance.setValues(
                mapping_params.keys(), mapping_params.values()
            )
            # reset is necessary to apply the changes to initial assignments
            roadrunner_instance.reset()
        if filling_mode == "only_species" or filling_mode == "all":
            # set species
            roadrunner_instance.setValues(
                mapping_species.keys(), mapping_species.values()
            )

    def fill_simulation_df(self, sim_res: Dict, edata: ExpData):
        """Fill a dataframe with the simulation results.

        Parameters
        ----------
        sim_res:
            Simulation results.
        edata:
            ExpData object.

        Returns
        -------
        sim_res_df:
            DataFrame with the simulation results.
        """
        sim_res_df = copy.deepcopy(edata.measurement_df)
        # in each row, replace the "measurement" with the simulation value
        for index, row in sim_res_df.iterrows():
            timepoint = row["time"]
            observable_id = row["observableId"]
            time_index = np.where(sim_res["time"] == timepoint)[0][0]
            sim_value = sim_res[observable_id][time_index]
            sim_res_df.at[index, "measurement"] = sim_value
        # rename measurement to simulation
        sim_res_df = sim_res_df.rename(columns={"measurement": "simulation"})
        return sim_res_df
