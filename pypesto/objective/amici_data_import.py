import numpy as np
import pandas as pd
import os
import sys
import importlib
import libsbml
import copy
import numbers
import amici
from pypesto.objective import AmiciObjective
from pypesto.problem import Problem


class Importer:

    def __init__(self, folder, output_folder=None, force_compile=False):
        self.dir = os.path.abspath(folder)

        self.name = os.path.split(self.dir)[-1]

        if output_folder is None:
            output_folder = os.path.abspath(os.path.join("tmp", self.name))
        self.output_folder = output_folder

        self.condition_file = os.path.join(self.dir,
            "experimentalCondition_" + self.name + ".tsv")
        self.measurement_file = os.path.join(self.dir,
            "measurementData_" + self.name + ".tsv")
        self.sbml_model_file = os.path.join(self.dir,
            "model_" + self.name + ".xml")
        self.parameter_file = os.path.join(self.dir,
            "parameters_" + self.name + ".tsv")
        
        self.condition_df = pd.read_csv(self.condition_file, sep='\t')
        self.measurement_df = pd.read_csv(self.measurement_file, sep='\t')
        self.parameter_df = pd.read_csv(self.parameter_file, sep='\t')
        
        self.model = None
        self.import_model(force_compile)
        
        self.solver = self.model.getSolver()
        
        self.x_nominal = None
        self.lb = None
        self.ub = None
        self.process_parameters()
            
    def import_model(self, force_compile=False):
        # compile
        if not os.path.exists(self.output_folder) or force_compile:
            self.compile_model()
        # add module to path
        if self.output_folder not in sys.path:
            sys.path.insert(0, self.output_folder)
        
        # load moduĺe
        model_module = importlib.import_module(self.name)
        
        # import model
        self.model = model_module.getModel()        
        
    def compile_model(self):
        # delete output directory
        if os.path.exists(self.output_folder):
            os.rmtree(self.output_folder)
        
        # init sbml importer
        sbml_importer = amici.SbmlImporter(self.sbml_model_file)
        
        # constant parameters
        constant_parameter_ids = self.condition_df.columns.values.tolist()[2:]
        
        # observables
        observables = amici.assignmentRules2observables(
            sbml_importer.sbml,
            filter_function=lambda v: v.getId().startswith('observable_')
        )

        # noise parameters
        observable_ids = list(observables.keys())
        sigmas = {}
        for observable_id in observable_ids:
            sigma_id = observable_id.replace("observable_", "sigma_", 1)
            formula = sbml_importer.sbml.getAssignmentRuleByVariable(
                sigma_id).getFormula()
            if observable_id not in sigmas:
                sigmas[observable_id] = formula
            elif sigmas[observable_id] != formula:
                raise ValueError(f"Inconsistent sigma specified for {observable_id}: "
                                 f"Previously {sigmas[observable_id]}, now {formula}")
        print(sigmas)
        # convert
        sbml_importer.sbml2amici(
            modelName=self.name,
            output_dir=self.output_folder,
            observables=observables,
            constantParameters=constant_parameter_ids,
            sigmas=sigmas
        )
        
    def process_parameters(self):
        self.par_ids = list(self.parameter_df['parameterId'])
        self.par_nominal_values = list(self.parameter_df['nominalValue'])
        self.par_lb = list(self.parameter_df['lowerBound'])
        self.par_ub = list(self.parameter_df['upperBound'])
        self.par_scale = list(self.parameter_df['parameterScale'])
        estimated = list(self.parameter_df['estimated'])
        self.par_fixed_indices = [j
                                  for j, val in enumerate(estimated)
                                  if val == 0]
        self.par_fixed_vals = [self.par_nominal_values[j]
                               for j, val in enumerate(estimated)
                               if val == 0]
        # TODO: update model par scales via model.setParameterScale()
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
        # conditions
        
        # number of amici simulations will be number of unique
        # (preequilibrationConditionId, simulationConditionId) pairs.
        # Can be improved by checking for identical condition vectors.
        
        # cannot group by nans
        
        conditions = self.measurement_df.groupby(
            [self.measurement_df['preequilibrationConditionId'].fillna("####"),
             self.measurement_df['simulationConditionId'].fillna("####")])\
             .size().reset_index()\
             .replace({'preequilibrationConditionId': {"####": np.nan}})\
             .replace({'simulationConditionId': {"####": np.nan}})
        
        observable_ids = self.model.getObservableIds()
        fixed_parameter_ids = self.model.getFixedParameterIds()
        
        edatas = []
        for _, condition in conditions.iterrows():
            # amici.ExpData for each simulation
            cur_measurement_df = self.measurement_df.loc[
                (self.measurement_df.preequilibrationConditionId == condition.preequilibrationConditionId) &
                (self.measurement_df.simulationConditionId == condition.simulationConditionId), :]
            
            timepoints = [ float(j) for j in sorted(cur_measurement_df.time.unique()) ]
            
            edata = amici.ExpData(self.model.get())
            edata.setTimepoints(timepoints)
            
            if len(fixed_parameter_ids) > 0:
                fixed_parameter_vals = self.condition_df.loc[
                    self.condition_df.conditionId == condition.simulationConditionId, fixed_parameter_ids].values
                edata.fixedParameters = fixed_parameter_vals.astype(float).flatten()
                
                if condition.preequilibrationConditionId is not np.nan:
                    fixed_preequilibration_parameter_vals = self.condition_df.loc[
                        self.condition_df.conditionId == condition.preequilibrationConditionId, fixed_parameter_ids].values
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
        
        obj = AmiciObjective(amici_model=self.model, amici_solver=self.solver, edata=edatas)
        
        return obj, edatas

    def create_problem(self, objective):
        problem = Problem(objective=objective,
                          lb=self.par_lb, ub=self.par_ub,
                          x_fixed_indices=self.par_fixed_indices,
                          x_fixed_vals=self.par_fixed_vals,
                          x_names=self.par_ids)

        return problem
