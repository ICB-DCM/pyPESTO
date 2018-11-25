import numpy as np
import pandas as pd
import os
import sys
import importlib
import libsbml
import copy
import amici
from .amici_objective import AmiciObjective


class Importer:

    def __init__(self, folder, output_folder=None):
        self.dir = os.path.abspath(folder)

        self.name = os.path.split(self.dir)[-1]

        if output_folder is not None:
            output_folder = os.path.join(self.dir, "tmp", self.name)
        self.output_folder = output_folder

        self.condition_file = os.path.join(self.dir,
            "experimentalCondition_" + self.name + ".tsv")
        self.measurement_file = os.path.join(self.dir,
            "measurementData_" + self.name + ".tsv")
        self.sbml_model_file = os.path.join(self.dir,
            "model_" + self.name + "_l2v4" + ".xml")
        
        self.condition_df = pd.read_csv(condition_file, sep='\t')
        self.measurement_df = pd.read_csv(measurement_file, sep='\t')

        self.model = None
        self.observables = None
        self.constant_parameter_ids = None
        self.parameter_ids = None
        self.parameter_names = None
        
        self.import_model()
            
    def import_model(self):
        if not os.path.exists(self.output_folder):
        self.compile_model()
        
    def compile_model(self):
        sbml_importer = amici.SbmlImporter(self.sbml_model_file)
        
        # constant parameters
        constant_parameter_ids = self.condition_df.columns.values.tolist()[2:]
        
        # observables
        observables = amici.assignmentRules2observables(
            sbml_importer.sbml,
            filter_function=lambda v: v.getId().startswith("observable_")
        )

        # noise parameters
        sigmas = {}
        df = copy.deepcopy(measurement_df)
        df.loc[df.noiseParameters.apply(isinstance, args=(float,)), "noiseParameters"] = np.nan
        obs_df = df.groupby(["observableId", "noiseParameters"]).size().reset_index()
        
        for _, row in obs_df.iterrows():
            assignment_rule = sbml_importer.sbml.getAssignmentRuleByVariable(
                f"sigma_{row.observableId}").getFormula()
            if assignment_rule in sigmas:
                sigmas[assignment_rule].add(row.observableId)
            else:
                sigmas[assignment_rule] = set([row.observableId])

        sbml_importer.sbml2amici(
            modelName=self.name,
            output_dir=self.output_folder,
            observables=observables,
            constantParameters=constant_parameter_ids,
            sigmas=sigmas
        )