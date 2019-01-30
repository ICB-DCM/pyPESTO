import numpy as np
import pandas as pd
import os
import sys
import importlib
import numbers
import copy

import amici
import petab
from pypesto.objective import AmiciObjective
from pypesto.problem import Problem


class PetabImporter:

    def __init__(self, petab_problem, output_folder=None, force_compile=False):
        """
        petab_problem: petab.Problem
            Managing access to the model and data.

        output_folder: str,  optional
            Folder to contain the amici model. Defaults to
            './tmp/petab_problem.name'.

        force_compile: str, optional
            If False, the model is compiled only if the output folder does not
            exist yet. If True, the output folder is deleted and the model
            (re-)compiled in either case.
        """
        self.petab_problem = petab_problem

        if output_folder is None:
            output_folder = os.path.abspath(
                os.path.join("tmp", self.petab_problem.model_name))
        self.output_folder = output_folder

        self.model = None
        self.import_model(force_compile)

        self.solver = self.model.getSolver()

    @staticmethod
    def from_folder(folder, output_folder=None, force_compile=False):
        """
        Simplified constructor exploiting the standardized petab folder
        structure.

        Parameters
        ----------

        folder: str
            Path to the base folder of the model, as in
            petab.Problem.from_folder.

        output_folder, force_compile: see __init__.
        """
        petab_problem = petab.Problem.from_folder(folder)

        return PetabImporter(
            petab_problem=petab_problem,
            output_folder=output_folder,
            force_compile=force_compile
        )

    def import_model(self, force_compile=False):
        """
        Import amici model. If necessary or force_compile is True, compile
        first.
        """
        # compile
        if not os.path.exists(self.output_folder) or force_compile:
            self.compile_model()

        # add module to path
        if self.output_folder not in sys.path:
            sys.path.insert(0, self.output_folder)

        # load moduĺe
        model_module = importlib.import_module(self.petab_problem.model_name)

        # import model
        self.model = model_module.getModel()

    def compile_model(self):
        """
        Compile the model. If the output folder exists already, it is first
        deleted.
        """

        # check prerequisites
        if not petab.check_condition_table_is_parameter_free(
                self.petab_problem.sbml_file):
            raise AssertionError(
                "Parameter dependent conditions in the condition file "
                "are not yet supported.")

        # delete output directory
        if os.path.exists(self.output_folder):
            os.rmtree(self.output_folder)

        # init sbml importer
        sbml_importer = amici.SbmlImporter(self.petab_problem.sbml_file)

        # constant parameters
        condition_columns = self.petab_problem.condition_df.columns.values
        constant_parameter_ids = list(
            set(condition_columns) - {'conditionId', 'conditionName'}
        )

        # observables
        observables = petab.get_observables(sbml_importer.sbml)

        # sigmas
        sigmas = petab.get_sigmas(sbml_importer.sbml)

        # convert
        sbml_importer.sbml2amici(
            modelName=self.petab_problem.model_name,
            output_dir=self.output_folder,
            observables=observables,
            constantParameters=constant_parameter_ids,
            sigmas=sigmas
        )

    def create_objective(self):
        """
        Create a pypesto.AmiciObjective.
        """

        condition_df = self.petab_problem.condition_df.reset_index()
        measurement_df = self.petab_problem.measurement_df

        # number of amici simulations will be number of unique
        # (preequilibrationConditionId, simulationConditionId) pairs.
        # Can be improved by checking for identical condition vectors.
        grouping_cols, simulation_conditions = \
            _get_simulation_conditions(condition_df, measurement_df)

        observable_ids = self.model.getObservableIds()

        fixed_parameter_ids = self.model.getFixedParameterIds()

        edatas = []
        for _, condition in simulation_conditions.iterrows():
            # amici.ExpData for each simulation

            # extract rows for condition
            cur_measurement_df = _get_rows_for_condition(
                measurement_df, grouping_cols, condition)

            # make list of all timepoints for which measurements exist
            timepoints = sorted(
                cur_measurement_df.time.unique().astype(float))

            # init edata object
            edata = amici.ExpData(self.model.get())
            edata.setTimepoints(timepoints)

            # handle fixed parameters
            _handle_fixed_parameters(
                edata, condition_df,
                fixed_parameter_ids, fixed_parameter_vals,
                condition
            )
            
            # prepare measurement matrix
            y = np.full(shape=(edata.nt(), edata.nytrue()), fill_value=np.nan)
            # prepare sigma matrix
            sigma_y = np.full(
                shape=(edata.nt(), edata.nytrue()),
                fill_value=np.nan)

            # add measurements and sigmas
            for _, measurement in cur_measurement_df.iterrows():
                time_ix = timepoints.index(measurement.time)
                observable_ix = observable_ids.index(
                    f'observable_{measurement.observableId}')

                y[time_ix, observable_ix] = measurement.measurement
                if isinstance(measurement.noiseParameters, numbers.Number):
                    sigma_y[time_ix, observable_ix] = \
                        measurement.noiseParameters

            # fill measurements and sigmas into edata
            edata.setObservedData(y.flatten())
            edata.setObservedDataStdDev(sigma_y.flatten())

            # append edata to edatas list
            edatas.append(edata)

        # simulation <-> optimization parameter mapping
        par_opt_ids = self.petab_problem.get_optimization_parameters()
        # take sim parameter vector from model to ensure correct order
        par_sim_ids = list(self.model.getParameterIds())
        # par_sim_ids = self.petab_problem.get_dynamic_simulation_parameters()

        parameter_mapping = \
            petab.core.get_optimization_to_simulation_parameter_mapping(
                condition_df=self.petab_problem.condition_df,
                measurement_df=self.petab_problem.measurement_df,
                parameter_df=self.petab_problem.parameter_df,
                sbml_model=self.petab_problem.sbml_model,
                par_opt_ids=par_opt_ids,
                par_sim_ids=par_sim_ids
            )

        scale_mapping = \
            petab.core.get_optimization_to_simulation_scale_mapping(
                parameter_df=self.petab_problem.parameter_df,
                mapping_par_opt_to_par_sim=parameter_mapping
            )

        # create objective
        obj = AmiciObjective(
            amici_model=self.model, amici_solver=self.solver, edata=edatas,
            x_ids=par_opt_ids, x_names=par_opt_ids,
            mapping_par_opt_to_par_sim=parameter_mapping,
            mapping_scale_opt_to_scale_sim=scale_mapping
        )

        return obj, edatas
    
    def create_problem(self, objective):
        problem = Problem(objective=objective,
                          lb=self.petab_problem.lb,
                          ub=self.petab_problem.ub,
                          x_fixed_indices=self.petab_problem.x_fixed_indices,
                          x_fixed_vals=self.petab_problem.x_fixed_vals,
                          x_names=self.petab_problem.x_ids)

        return problem

    def rdatas_to_measurement_df(self, rdatas):
        """
        Create a measurement dataframe in the petab format from
        the passed `rdatas` and own information.

        Parameters
        ----------

        rdatas: list of amici.RData
            A list of rdatas as produced by
            pypesto.AmiciObjective.__call__(x, return_dict=True)['rdatas'].

        Returns
        -------

        df: pandas.DataFrame
            A dataframe built from the rdatas in the format as in
            self.petab_problem.measurement_df.
        """

        condition_df = self.petab_problem.condition_df.reset_index()
        measurement_df = self.petab_problem.measurement_df

        # initialize dataframe
        df = pd.DataFrame(
            columns=list(
                self.petab_problem.measurement_df.columns))

        # get simulation conditions
        grouping_cols, simulation_conditions = \
            _get_simulation_conditions(condition_df, measurement_df)

        # get observable ids
        observable_ids = self.model.getObservableIds()

        # iterate over conditions
        for data_idx, condition in simulation_conditions.iterrows():
            # current rdata
            rdata = rdatas[data_idx]
            # current simulation matrix
            y = rdata['y']
            # time array used in rdata
            t = list(rdata['t'])

            # extract rows for condition
            cur_measurement_df = _get_rows_for_condition(
                measurement_df, grouping_cols, condition)

            # iterate over entries for the given condition
            # note: this way we only generate a dataframe entry for every
            # row that existed in the original dataframe. if we want to
            # e.g. have also timepoints non-existent in the original file,
            # we need to instead iterate over the rdata['y'] entries
            for _, row in cur_measurement_df.iterrows():
                # copy row
                row_sim = copy.deepcopy(row)

                # extract simulated measurement value
                timepoint_idx = t.index(row.time)
                observable_idx = observable_ids.index(
                    "observable_" + row.observableId)
                measurement_sim = y[timepoint_idx, observable_idx]

                # change measurement entry
                row_sim.measurement = measurement_sim

                # append to dataframe
                df = df.append(row_sim, ignore_index=True)

        return df


def _get_simulation_conditions(condition_df, measurement_df):
    """
    Compute the conditions by which to group the measurements, so
    that for each group an amici.EData can be generated.

    Returns
    -------
    (grouping_cols, simulation_conditions): tuple
        Here, grouping_cols is the columns according to which the grouping
        was done, and simulation_conditions are the identified conditions.
    """
    # make sure index is reset
    condition_df = condition_df.reset_index()

    # find columns to group by (i.e. if not all nans).
    # number of amici simulations will be number of unique
    # (preequilibrationCondition, simulationCondition) pairs.
    # can be improved by checking for identical condition vectors.
    grouping_cols = petab.core.get_notnull_columns(
        measurement_df,
        ['simulationConditionId', 'preequilibrationConditionId'])

    # group by cols and return dataframe containing each combination
    # of those rows only once (and an additional counting row)
    simulation_conditions = measurement_df.groupby(
        grouping_cols).size().reset_index()

    return grouping_cols, simulation_conditions


def _get_rows_for_condition(measurement_df, grouping_cols, condition):
    """
    Extract rows in `measurement_df` according to `grouping_cols`
    for `condition`.

    Returns
    -------

    cur_measurement_df: pd.DataFrame
        The subselection for the condition.
    """
    # filter rows for condition
    row_filter = 1
    # check for equality in all grouping cols
    for col in grouping_cols:
        row_filter = (
            measurement_df[col] == condition[col]
        ) & row_filter

    # apply filter
    cur_measurement_df = measurement_df.loc[row_filter, :]
    return cur_measurement_df


def _handle_fixed_parameters(
        edata, condition_df,
        fixed_parameter_ids, fixed_parameter_vals,
        condition):
    """
    Hande fixed parameters and update edata accordingly.
    """

    if len(fixed_parameter_ids) == 0:
        # nothing to be done
        return

    fixed_parameter_vals = condition_df.loc[
        condition_df.conditionId ==
        condition.simulationConditionId,
        fixed_parameter_ids].values
    edata.fixedParameters = fixed_parameter_vals.astype(
        float).flatten()

    if 'preequilibrationConditionId' in condition \
            and condition.preequilibrationConditionId:
        fixed_preequilibration_parameter_vals = condition_df.loc[
            # TODO: preequilibrationConditionId might not exist
            condition_df.conditionId == \
            condition.preequilibrationConditionId,
            fixed_parameter_ids].values
        edata.fixedParametersPreequilibration = \
            fixed_preequilibration_parameter_vals.astype(float) \
                                                 .flatten()

