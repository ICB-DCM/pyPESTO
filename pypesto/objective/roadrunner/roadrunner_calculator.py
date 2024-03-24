import numbers
from typing import Optional, Sequence

import numpy as np
import petab
import roadrunner
from petab.parameter_mapping import ParMappingDictQuadruple

from ...C import MODE_FUN, MODE_RES, ModeType
from .utils import ExpData, unscale_parameters

LLH_TYPES = {
    "lin_normal": lambda x, y, z: -0.5
    * (np.log(2 * np.pi * (z**2)) + ((x - y) / z) ** 2),
    "log_normal": lambda x, y, z: -0.5
    * (
        np.log(2 * np.pi * (z**2) * (x**2))
        + ((np.log(x) - np.log(y)) / z) ** 2
    ),
    "log10_normal": lambda x, y, z: -0.5
    * (
        np.log(2 * np.pi * (z**2) * (x**2) * np.log(10) ** 2)
        + ((np.log10(x) - np.log10(y)) / z) ** 2
    ),
    "lin_laplace": lambda x, y, z: -np.log(2 * z) - (np.abs(x - y) / z),
    "log_laplace": lambda x, y, z: -np.log(2 * z * y)
    - (np.abs(np.log(x) - np.log(y)) / z),
    "log10_laplace": lambda x, y, z: -np.log(2 * z * y * np.log(10))
    - (np.abs(np.log10(x) - np.log10(y)) / z),
}


class RoadRunnerCalculator:
    """Class to handle RoadRunner simulation and obtain objective value."""

    def __init__(self):
        pass

    def __call__(
        self,
        x_dct: dict,  # TODO: sensi_order support
        mode: ModeType,
        roadrunner_instance: roadrunner.RoadRunner,
        edatas: list[ExpData],
        x_ids: Sequence[str],
        parameter_mapping: list[ParMappingDictQuadruple],
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
        petab_problem:
            PEtab problem.

        Returns
        -------
        Tuple of objective function values.
        """
        # sanitiy check that edatas and conditions are consistent
        if len(edatas) != len(parameter_mapping):
            raise ValueError(
                "Number of edatas and conditions are not consistent."
            )
        simulation_results = {}
        llh_tot = 0
        for edata, mapping_per_condition in zip(edatas, parameter_mapping):
            sim_res, llh = self.simulate_per_condition(
                x_dct, roadrunner_instance, edata, mapping_per_condition
            )
            simulation_results[edata.condition_id] = sim_res
            llh_tot += llh

        if mode == MODE_FUN:
            return {
                "fval": -llh_tot,
                "simulation_results": simulation_results,
                "llh": llh_tot,
            }
        if mode == MODE_RES:
            res_df = petab.calculate_residuals(
                petab_problem.measurement_df,
                simulation_results,
                petab_problem.observable_df,
                petab_problem.parameter_df,
            )
            return {
                "res": res_df,
                "simulation_results": simulation_results,
                "fval": -llh_tot,
            }

    def simulate_per_condition(
        self,
        x_dct: dict,
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
        if timepoints[0] != 0.0:
            timepoints = [0.0] + timepoints
        if len(timepoints) == 1:
            timepoints = [0.0] + timepoints
        observables_ids = edata.get_observable_ids()
        # steady state stuff
        steady_state_calculations = False
        state_variables = roadrunner_instance.model.getFloatingSpeciesIds()
        # obs_ss = []  # TODO: add them to return values with info
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
            # obs_ss = steady_state[:, : len(observables_ids)].flatten()
            state_ss = steady_state[:, len(observables_ids) :].flatten()
            # turn off conserved moiety analysis
            roadrunner_instance.conservedMoietyAnalysis = False
            # reset the model
            roadrunner_instance.reset()
        # set parameters
        par_map = self.fill_in_parameters(
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

        llhs = calculate_llh(sim_res, edata, par_map)

        # reset the model
        roadrunner_instance.reset()

        return sim_res, llhs

    def fill_in_parameters(
        self,
        problem_parameters: dict,
        roadrunner_instance: Optional[roadrunner.RoadRunner] = None,
        parameter_mapping: Optional[ParMappingDictQuadruple] = None,
        preeq: bool = False,
        filling_mode: Optional[str] = None,
    ) -> dict:
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

        Returns
        -------
        dict:
            Mapping of parameter IDs to values.
        """
        if filling_mode is None:
            filling_mode = "all"
        mapping = parameter_mapping[1]
        scaling = parameter_mapping[3]
        if preeq:
            mapping = parameter_mapping[0]
            scaling = parameter_mapping[2]

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
        mapping_params = {}
        mapping_species = {}
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
        return mapping_values


def calculate_llh(
    simulations: np.ndarray,
    edata: ExpData,
    parameter_mapping: dict,
) -> float:
    """Calculate the negative log-likelihood. for a single condition.

    Parameters
    ----------
    simulations:
        Simulations of condition.
    edata:
        ExpData of a single condition.

    Returns
    -------
    float:
        Negative log-likelihood.
    """
    # if 0 is not in timepoints, remove the first row of the simulation
    if 0.0 not in edata.timepoints:
        simulations = simulations[1:, :]

    def _fill_simulation_w_replicates(simulations, measurements) -> np.ndarray:
        """Fill the simulation with replicates.

        Parameters
        ----------
        simulations:
            Simulations, without replicates.
        measurements:
            Measurements, with replicates.

        Returns
        -------
        np.ndarray:
            An array of simulations where each row has its peaundant in the
            measurements. Replicates in measurements result in copies of the
            corresponding simulation.
        """
        # Find unique time values in measurements
        unique_time_values = np.unique(measurements[:, 0])

        # Initialize an empty list to store the replicated rows
        replicated_rows = []

        # Iterate over unique time values
        for time_value in unique_time_values:
            # Find the rows in measurements with the current time value
            matching_rows = measurements[measurements[:, 0] == time_value]
            # Append replicated rows from simulations for each matching row in measurements
            replicated_rows.extend(
                [
                    row_sim
                    for row_sim in simulations[simulations[:, 0] == time_value]
                    for _ in range(len(matching_rows))
                ]
            )

        # Convert the list of replicated rows to a NumPy array
        replicated_simulations = np.array(replicated_rows)

        return replicated_simulations

    if not np.array_equal(simulations[:, 0], edata.timepoints):
        raise ValueError(
            "Simulation and Measurement have different timepoints."
        )
    # if timepoints in measurements and simulations are not the same, fill
    if len(simulations[:, 0]) != len(edata.measurements[:, 0]):
        simulations = _fill_simulation_w_replicates(
            simulations, edata.measurements
        )
    # check that simulation and condition have same dimensions and timepoints
    if simulations.shape != edata.measurements.shape:
        raise ValueError(
            "Simulation and Measurement have different dimensions."
        )
    # we can now drop the timepoints
    simulations = simulations[:, 1:]
    measurements = edata.measurements[:, 1:]

    def _fill_in_noise_formula(noise_formula):
        """Fill in the noise formula."""
        if isinstance(noise_formula, numbers.Number):
            return float(noise_formula)
        # if it is not a number, it is assumed to be a string
        if noise_formula in parameter_mapping.keys():
            return parameter_mapping[noise_formula]

    # replace noise formula with actual value from mapping
    noise_formulae = np.array(
        [_fill_in_noise_formula(formula) for formula in edata.noise_formulae]
    )
    # check that the rows of noise are the columns of the simulation
    if noise_formulae.shape[0] != simulations.shape[1]:
        raise ValueError("Noise and Simulation have different dimensions.")
    # # duplicate the noise formulae to match the number of rows of the simulation
    # noise_formulae = np.tile(noise_formulae, (simulations.shape[0], 1))
    # # do the same for the noise distributions
    # noise_dist = np.tile(edata.noise_distributions, (simulations.shape[0], 1))
    # per observable, decide on the llh function based on the noise dist
    llhs = [
        LLH_TYPES[noise_dist](
            measurements[:, i], simulations[:, i], noise_formulae[i]
        )
        for i, noise_dist in enumerate(edata.noise_distributions)
    ]
    # sum over all observables
    llhs = np.sum(llhs)

    return llhs
