import numpy as np
import pandas as pd
import os
import sys
import importlib
import libsbml
import copy
import numbers

import amici
import petab
from pypesto.objective import AmiciObjective
from pypesto.problem import Problem


class Importer:

    def __init__(self, petab_manager, output_folder=None, force_compile=False):
        
        self.petab_manager = petab_manager

        if output_folder is None:
            output_folder = os.path.abspath(os.path.join("tmp", self.petab_manager.name))
        self.output_folder = output_folder
        
        self.model = None
        self.import_model(force_compile)
        
        self.solver = self.model.getSolver()
        
        self.x_nominal = None
        self.lb = None
        self.ub = None
        #self.process_parameters()

    @staticmethod
    def from_folder(folder, output_folder=None, force_compile=False):
        petab_manager = petab.Manager.from_folder(folder)

        return Importer(
            petab_manager=petab_manager,
            output_folder=output_folder,
            force_compile=force_compile
        )

    def import_model(self, force_compile=False):
        # compile
        if not os.path.exists(self.output_folder) or force_compile:
            self.compile_model()
        # add module to path
        if self.output_folder not in sys.path:
            sys.path.insert(0, self.output_folder)
        
        # load moduĺe
        model_module = importlib.import_module(self.petab_manager.name)
        
        # import model
        self.model = model_module.getModel()        
        
    def compile_model(self):
        # delete output directory
        if os.path.exists(self.output_folder):
            os.rmtree(self.output_folder)
        
        # init sbml importer
        sbml_importer = amici.SbmlImporter(self.petab_manager.sbml_file)
        
        # constant parameters
        constant_parameter_ids = list(
            set(self.petab_manager.condition_df.columns.values) - {'conditionId', 'conditionName'}
        )

        # observables
        observables = petab.get_observables(sbml_importer.sbml)

        # sigmas
        sigmas = petab.get_sigmas(sbml_importer.sbml)
        sigmas = {key.replace('sigma_', 'observable_', 1) : value['formula'] for key, value in sigmas.items()}

        # convert
        sbml_importer.sbml2amici(
            modelName=self.petab_manager.name,
            output_dir=self.output_folder,
            observables=observables,
            constantParameters=constant_parameter_ids,
            sigmas=sigmas
        )
        
    def process_parameters(self):
        parameter_df = self.petab_manager.parameter_df.reset_index()
        self.par_ids = list(parameter_df['parameterId'])
        self.par_nominal_values = list(parameter_df['nominalValue'])
        self.par_lb = list(parameter_df['lowerBound'])
        self.par_ub = list(parameter_df['upperBound'])
        self.par_scale = list(parameter_df['parameterScale'])
        estimated = list(parameter_df['estimate'])
        self.par_fixed_indices = [j for j, val in enumerate(estimated)
                                  if val == 0]
        self.par_fixed_vals = [self.par_nominal_values[j]
                               for j, val in enumerate(estimated)
                               if val == 0]
        scaling_vector = amici.ScalingVector()
        for scale_str in self.par_scale:
            if scale_str == 'lin':
                scale = amici.ParameterScaling_none
            elif scale_str == 'log10':
                scale = amici.ParameterScaling_log10
            elif scale_str == 'log':
                scale = amici.ParameterScaling_ln
            else:
                raise ValueError(f"Parameter scaling not recognized: {scale_str}")
            scaling_vector.append(scale)
        self.model.setParameterScale(scaling_vector)
        
    def create_objective(self):
        # number of amici simulations will be number of unique
        # (preequilibrationConditionId, simulationConditionId) pairs.
        # Can be improved by checking for identical condition vectors.
        
        condition_df = self.petab_manager.condition_df
        measurement_df = self.petab_manager.measurement_df

        grouping_cols = petab.core.get_notnull_columns(
            measurement_df, ['simulationConditionId', 'preequilibrationConditionId'])
        simulation_conditions = measurement_df.groupby(grouping_cols).size().reset_index()
        
        observable_ids = self.model.getObservableIds()
        fixed_parameter_ids = self.model.getFixedParameterIds()
        
        edatas = []
        for edata_idx, condition in simulation_conditions.iterrows():
            # amici.ExpData for each simulation
            filter = 1
            for col in grouping_cols:
                filter = (measurement_df[col] == condition[col]) & filter
            cur_measurement_df = measurement_df.loc[filter, :]
            print('CUR',cur_measurement_df) 
            timepoints = sorted(cur_measurement_df.time.unique().astype(float))
            
            edata = amici.ExpData(self.model.get())
            edata.setTimepoints(timepoints)
            
            if len(fixed_parameter_ids) > 0:
                fixed_parameter_vals = condition_df.loc[
                    condition_df.conditionId == condition.simulationConditionId, fixed_parameter_ids].values
                edata.fixedParameters = fixed_parameter_vals.astype(float).flatten()
                print('FP',edata.fixedParameters)
                if 'preequilibrationConditionId' in condition and condition.preequilibrationConditionId:
                    fixed_preequilibration_parameter_vals = condition_df.loc[
                        # TODO: preequilibrationConditionId might not exist
                        condition_df.conditionId == condition.preequilibrationConditionId, fixed_parameter_ids].values
                    edata.fixedParametersPreequilibration = fixed_preequilibration_parameter_vals.astype(float).flatten()
            
            y = np.full(shape=(edata.nt(), edata.nytrue()), fill_value=np.nan)
            sigma_y = np.full(shape=(edata.nt(), edata.nytrue()), fill_value=np.nan)
            
            # add measurements and sigmas
            for _, measurement in cur_measurement_df.iterrows():
                time_ix = timepoints.index(measurement.time)
                observable_ix = observable_ids.index(f'observable_{measurement.observableId}')
                # TODO: measurement file should contain prefix
                
                y[time_ix, observable_ix] = measurement.measurement
                if isinstance(measurement.noiseParameters, numbers.Number):
                    sigma_y[time_ix, observable_ix] = measurement.noiseParameters
            
            edata.setObservedData(y.flatten())
            edata.setObservedDataStdDev(sigma_y.flatten())
            edatas.append(edata)
        
        # simulation <-> optimization parameter mapping
        par_opt_ids = self.petab_manager.get_optimization_parameters()
        par_sim_ids = self.petab_manager.get_dynamic_simulation_parameters()
        
        mapping = petab.core.map_par_sim_to_par_opt(
            condition_df=self.petab_manager.condition_df,
            measurement_df=self.petab_manager.measurement_df,
            parameter_df=self.petab_manager.parameter_df,
            sbml_model=self.petab_manager.sbml_model,
            par_opt_ids=par_opt_ids,
            par_sim_ids=par_sim_ids
        )
        
        # create objective
        obj = AmiciObjective(
            amici_model=self.model, amici_solver=self.solver, edata=edatas,
            par_opt_ids=par_opt_ids, par_sim_ids=par_sim_ids, mapping=mapping
        )
        
        return obj, edatas

    def create_problem(self, objective):
        problem = Problem(objective=objective,
                          lb=self.par_lb, ub=self.par_ub,
                          x_fixed_indices=self.par_fixed_indices,
                          x_fixed_vals=self.par_fixed_vals,
                          x_names=self.par_ids)

        return problem
