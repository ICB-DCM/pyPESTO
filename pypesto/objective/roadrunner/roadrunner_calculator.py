"""RoadRunner calculator for PEtab problems.

Handles all RoadRunner.simulate calls, calculates likelihoods and residuals.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import numpy as np

from ...C import (
    FVAL,
    MODE_FUN,
    MODE_RES,
    RES,
    ROADRUNNER_LLH,
    ROADRUNNER_SIMULATION,
    TIME,
    ModeType,
)
from .utils import (
    ExpData,
    SolverOptions,
    simulation_to_measurement_df,
    unscale_parameters,
)

try:
    import petab.v1 as petab
    from petab.v1.parameter_mapping import ParMappingDictQuadruple
except ImportError:
    petab = None
try:
    import roadrunner
except ImportError:
    roadrunner = None

LLH_TYPES = {
    "lin_normal": lambda measurement, simulation, sigma: -0.5
    * (
        np.log(2 * np.pi * (sigma**2))
        + ((measurement - simulation) / sigma) ** 2
    ),
    "log_normal": lambda measurement, simulation, sigma: -0.5
    * (
        np.log(2 * np.pi * (sigma**2) * (measurement**2))
        + ((np.log(measurement) - np.log(simulation)) / sigma) ** 2
    ),
    "log10_normal": lambda measurement, simulation, sigma: -0.5
    * (
        np.log(2 * np.pi * (sigma**2) * (measurement**2) * np.log(10) ** 2)
        + ((np.log10(measurement) - np.log10(simulation)) / sigma) ** 2
    ),
    "lin_laplace": lambda measurement, simulation, sigma: -np.log(2 * sigma)
    - (np.abs(measurement - simulation) / sigma),
    "log_laplace": lambda measurement, simulation, sigma: -np.log(
        2 * sigma * simulation
    )
    - (np.abs(np.log(measurement) - np.log(simulation)) / sigma),
    "log10_laplace": lambda measurement, simulation, sigma: -np.log(
        2 * sigma * simulation * np.log(10)
    )
    - (np.abs(np.log10(measurement) - np.log10(simulation)) / sigma),
}


class RoadRunnerCalculator:
    """Class to handle RoadRunner simulation and obtain objective value."""

    def __call__(
        self,
        x_dct: dict,  # TODO: sensi_order support
        mode: ModeType,
        roadrunner_instance: roadrunner.RoadRunner,
        edatas: list[ExpData],
        x_ids: Sequence[str],
        parameter_mapping: list[ParMappingDictQuadruple],
        petab_problem: petab.Problem,
        solver_options: SolverOptions | None = None,
    ):
        """Perform the RoadRunner call and obtain objective function values.

        Parameters
        ----------
        x_dct:
            Parameter dictionary.
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
        solver_options:
            Solver options of the roadrunner instance Integrator. These will
            modify the roadrunner instance inplace.

        Returns
        -------
        Tuple of objective function values.
        """
        # sanity check that edatas and conditions are consistent
        if len(edatas) != len(parameter_mapping):
            raise ValueError(
                "Number of edatas and conditions are not consistent."
            )
        if solver_options is None:
            solver_options = SolverOptions()
        # apply solver options
        solver_options.apply_to_roadrunner(roadrunner_instance)
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
                FVAL: -llh_tot,
                ROADRUNNER_SIMULATION: simulation_results,
                ROADRUNNER_LLH: llh_tot,
            }
        if mode == MODE_RES:  # TODO: speed up by not using pandas
            simulation_df = simulation_to_measurement_df(
                simulation_results, petab_problem.measurement_df
            )
            res_df = petab.calculate_residuals(
                petab_problem.measurement_df,
                simulation_df,
                petab_problem.observable_df,
                petab_problem.parameter_df,
            )
            return {
                RES: res_df,
                ROADRUNNER_SIMULATION: simulation_results,
                FVAL: -llh_tot,
            }

    def simulate_per_condition(
        self,
        x_dct: dict,
        roadrunner_instance: roadrunner.RoadRunner,
        edata: ExpData,
        parameter_mapping_per_condition: ParMappingDictQuadruple,
    ) -> tuple[np.ndarray, float]:
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

        Returns
        -------
        Tuple of simulation results in form of a numpy array and the
        negative log-likelihood.
        """
        # get timepoints and outputs to simulate
        timepoints = list(edata.timepoints)
        if timepoints[0] != 0.0:
            timepoints = [0.0] + timepoints
        if len(timepoints) == 1:
            timepoints = [0.0] + timepoints
        observables_ids = edata.get_observable_ids()
        # steady state stuff
        steady_state_calculations = False
        state_variables = roadrunner_instance.model.getFloatingSpeciesIds()
        # some states might be hidden as parameters with rate rules
        rate_rule_ids = roadrunner_instance.getRateRuleIds()
        state_variables += [
            rate_rule_id
            for rate_rule_id in rate_rule_ids
            if rate_rule_id not in state_variables
        ]
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
            roadrunner_instance.setSteadyStateSolver("newton")
            # allow simulation to reach steady state
            roadrunner_instance.getSteadyStateSolver().setValue(
                "allow_presimulation", True
            )
            roadrunner_instance.getSteadyStateSolver().setValue(
                "presimulation_maximum_steps", 1000
            )
            roadrunner_instance.getSteadyStateSolver().setValue(
                "presimulation_time", 1000
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
            times=timepoints, selections=[TIME] + observables_ids
        )

        llhs = calculate_llh(sim_res, edata, par_map, roadrunner_instance)

        # reset the model
        roadrunner_instance.reset()

        return sim_res, llhs

    def fill_in_parameters(
        self,
        problem_parameters: dict,
        roadrunner_instance: roadrunner.RoadRunner | None = None,
        parameter_mapping: ParMappingDictQuadruple | None = None,
        preeq: bool = False,
        filling_mode: str | None = None,
    ) -> dict:
        """Fill in parameters into the roadrunner instance.

        Parameters
        ----------
        roadrunner_instance:
            RoadRunner instance to fill in parameters
        problem_parameters:
            Problem parameters as parameterId=>value dict. Only
            parameters included here will be set. Remaining parameters will
            be used as already set in `amici_model` and `edata`.
        parameter_mapping:
            Parameter mapping for current condition. Quadruple of dicts,
            where the first dict contains the parameter mapping for pre-
            equilibration, the second dict contains the parameter mapping
            for the simulation, the third and fourth dict contain the scaling
            factors for the pre-equilibration and simulation, respectively.
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
        # check for valid filling modes
        if filling_mode not in ["all", "only_parameters", "only_species"]:
            raise ValueError(
                "Invalid filling mode. Choose from 'all', "
                "'only_parameters', 'only_species'."
            )
        mapping = parameter_mapping[1]  # default: simulation condition mapping
        scaling = parameter_mapping[3]  # default: simulation condition scaling
        if preeq:
            mapping = parameter_mapping[0]  # pre-equilibration mapping
            scaling = parameter_mapping[2]  # pre-equilibration scaling

        # Parameter parameter_mapping may contain parameter_ids as values,
        # these *must* be replaced

        def _get_par(model_par, val):
            """Get parameter value from problem_parameters and mapping.

            Replace parameter IDs in parameter_mapping dicts by values from
            problem_parameters where necessary
            """
            if isinstance(val, str):
                try:
                    # estimated parameter
                    return problem_parameters[val]
                except KeyError:
                    # condition table overrides must have been handled already,
                    # e.g. by the PEtab parameter parameter_mapping, but
                    # parameters from InitialAssignments may still be present.
                    if mapping[val] == model_par:
                        # prevent infinite recursion
                        raise
                    return _get_par(val, mapping[val])
            if model_par in problem_parameters:
                # user-provided
                return problem_parameters[model_par]
            # prevent nan-propagation in derivative
            if np.isnan(val):
                return 0.0
            # constant value
            return val

        mapping_values = {
            key: _get_par(key, val) for key, val in mapping.items()
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
            roadrunner_instance.setValues(mapping_params)
            # reset is necessary to apply the changes to initial assignments
            roadrunner_instance.reset()
        if filling_mode == "only_species" or filling_mode == "all":
            # set species
            roadrunner_instance.setValues(mapping_species)
        return mapping_values


def calculate_llh(
    simulations: np.ndarray,
    edata: ExpData,
    parameter_mapping: dict,
    roadrunner_instance: roadrunner.RoadRunner,
) -> float:
    """Calculate the negative log-likelihood for a single condition.

    Parameters
    ----------
    simulations:
        Simulations of condition.
    edata:
        ExpData of a single condition.
    parameter_mapping:
        Parameter mapping for the condition.
    roadrunner_instance:
        RoadRunner instance. Needed to retrieve complex formulae.

    Returns
    -------
    float:
        Negative log-likelihood.
    """
    # if 0 is not in timepoints, remove the first row of the simulation
    if 0.0 not in edata.timepoints:
        simulations = simulations[1:, :]

    if not np.array_equal(simulations[:, 0], edata.timepoints):
        raise ValueError(
            "Simulation and Measurement have different timepoints."
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
        # if the string starts with "noiseFormula_" it is saved in the model
        if noise_formula.startswith("noiseFormula_"):
            return roadrunner_instance.getValue(noise_formula)

    # replace noise formula with actual value from mapping
    noise_formulae = np.array(
        [_fill_in_noise_formula(formula) for formula in edata.noise_formulae]
    )
    # check that the rows of noise are the columns of the simulation
    if noise_formulae.shape[0] != simulations.shape[1]:
        raise ValueError("Noise and Simulation have different dimensions.")
    # per observable, decide on the llh function based on the noise dist
    llhs = np.array(
        [
            LLH_TYPES[noise_dist](
                measurements[:, i], simulations[:, i], noise_formulae[i]
            )
            for i, noise_dist in enumerate(edata.noise_distributions)
        ]
    ).transpose()
    # check whether all nan values in llhs coincide with nan measurements
    if not np.all(np.isnan(llhs) == np.isnan(measurements)):
        return np.nan

    # sum over all observables, ignoring nans
    llhs = np.nansum(llhs)

    return llhs
